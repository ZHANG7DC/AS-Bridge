from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from dataset import ZarrPairDataset
from model import BrownianBridge
from unet import UNet


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        if str(value).lower() == "none":
            return []
        return [str(value)]
    return [str(v) for v in value if v and str(v).lower() != "none"]


def _resolve_path(repo_root: Path, path: str | None) -> str | None:
    if not path:
        return None
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = repo_root / p
    return str(p)


def _default_group_name(path: Path) -> str:
    if path.suffix == ".zarr":
        name = path.stem
    else:
        name = path.name
    if name == "zarr":
        name = f"{path.parent.name}_{name}"
    return name.replace(".", "_")


def _build_truncated_steps(start_t: int, num_steps: int | None, num_timesteps: int) -> torch.Tensor:
    start_t = int(start_t)
    if start_t < 0:
        raise ValueError("start_t must be >= 0")
    if start_t >= int(num_timesteps):
        raise ValueError(f"start_t must be < num_timesteps={num_timesteps}")

    if num_steps is None:
        return torch.arange(start_t, -1, -1, dtype=torch.long)

    num_steps = int(num_steps)
    if num_steps <= 0:
        raise ValueError("sample_steps must be > 0 or None")

    num_steps = min(num_steps, start_t + 1)
    if start_t == 0:
        return torch.tensor([0], dtype=torch.long)
    if num_steps == 1:
        return torch.tensor([start_t, 0], dtype=torch.long)

    raw = torch.linspace(start_t, 0, num_steps).round().long()
    raw[0] = start_t
    raw[-1] = 0

    steps_list: list[int] = []
    last = None
    for s in raw.tolist():
        s = int(s)
        if last is None or s < last:
            steps_list.append(s)
            last = s
    if steps_list[-1] != 0:
        steps_list.append(0)
    return torch.tensor(steps_list, dtype=torch.long)


@torch.no_grad()
def _denoise_from_xt(model: BrownianBridge, x_t: torch.Tensor, y: torch.Tensor, steps: torch.Tensor, *, source, target, direction: int):
    old_steps = model.steps
    model.steps = steps
    img = x_t
    try:
        for i in range(len(steps)):
            img, _ = model.p_sample(
                img, y, i,
                clip_denoised=True,
                source=source,
                target=target,
                direction=direction,
            )
        return img
    finally:
        model.steps = old_steps


def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def _min_pixel_mse01(candidates: torch.Tensor, gt: torch.Tensor):
    """Compute per-sample mean of min per-pixel squared error in [0,1] space.

    candidates: (K,B,C,H,W)
    gt:         (B,C,H,W)
    returns:
      err_mean: (B,)
      min_err_map: (B,H,W)
    """
    cand01 = _to01(candidates)
    gt01 = _to01(gt)
    diffs = (cand01 - gt01.unsqueeze(0)).pow(2).mean(dim=2)  # (K,B,H,W)
    min_err_map = torch.min(diffs, dim=0).values  # (B,H,W)
    err_mean = min_err_map.mean(dim=(1, 2))
    return err_mean, min_err_map


def _min_pixel_mse01_norm(candidates: torch.Tensor, gt: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalized anomaly score: per-channel min-MSE / per-channel intensity."""
    cand01 = _to01(candidates)
    gt01 = _to01(gt)
    diffs = (cand01 - gt01.unsqueeze(0)).pow(2)  # (K,B,C,H,W)
    min_err = torch.min(diffs, dim=0).values  # (B,C,H,W)
    err_sum = min_err.sum(dim=(2, 3))  # (B,C)
    inten_sum = gt01.sum(dim=(2, 3))  # (B,C)
    norm = err_sum / (inten_sum + eps)
    return norm.mean(dim=1)


def _init_rng_states(seeds: list[int], device: torch.device):
    cpu_states: dict[int, torch.Tensor] = {}
    cuda_states: dict[int, list[torch.Tensor]] | None = None
    if device.type == "cuda":
        cuda_states = {}
    for seed in seeds:
        seed = int(seed)
        torch.manual_seed(seed)
        cpu_states[seed] = torch.get_rng_state()
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            assert cuda_states is not None
            cuda_states[seed] = torch.cuda.get_rng_state_all()
    return cpu_states, cuda_states


def _restore_rng_state(seed: int, cpu_states, cuda_states, device: torch.device):
    torch.set_rng_state(cpu_states[seed])
    if device.type == "cuda":
        assert cuda_states is not None
        torch.cuda.set_rng_state_all(cuda_states[seed])


def _save_rng_state(seed: int, cpu_states, cuda_states, device: torch.device):
    cpu_states[seed] = torch.get_rng_state()
    if device.type == "cuda":
        assert cuda_states is not None
        cuda_states[seed] = torch.cuda.get_rng_state_all()


class _IndexWrapper(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int] | None):
        self.dataset = dataset
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        if self.indices is None:
            return len(self.dataset)
        return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is None:
            real_idx = int(idx)
        else:
            real_idx = int(self.indices[int(idx)])
        item = self.dataset[real_idx]
        item["index"] = real_idx
        return item


def _get_fixed_indices(zarr_path: Path, n_samples: int, save_path: Path, key: str):
    import zarr

    z_arr = zarr.open(str(zarr_path), mode="r")
    total_count = int(z_arr.shape[0])

    indices = None
    if save_path.exists():
        with np.load(save_path) as data:
            if key in data:
                loaded = data[key]
                if len(loaded) == n_samples:
                    indices = loaded
                else:
                    print(f"[{key}] Size mismatch (saved {len(loaded)} vs need {n_samples}). Resampling.")
    if indices is None:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(total_count, size=n_samples, replace=False)
        if save_path.exists():
            data_dict = dict(np.load(save_path))
        else:
            data_dict = {}
        data_dict[key] = indices
        np.savez(save_path, **data_dict)
    return indices


def _build_shared_model(model_cfg: dict, device: torch.device) -> BrownianBridge:
    channels = int(model_cfg.get("channels", 3))
    net = UNet(
        in_channels=channels * 3,
        out_channels=channels * 2,
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        time_dim=int(model_cfg.get("time_dim", 256)),
    )
    return BrownianBridge(
        denoise_fn=net,
        num_timesteps=int(model_cfg.get("num_timesteps", 1000)),
        mt_type=str(model_cfg.get("mt_type", "linear")),
        max_var=float(model_cfg.get("max_var", 1.0)),
        eta=float(model_cfg.get("eta", 1.0)),
        objective=str(model_cfg.get("objective", "eps")),
        loss_type=str(model_cfg.get("loss_type", "mse")),
        channels=channels,
        nll_a_init=float(model_cfg.get("nll_a_init", 0.0)),
        nll_b_init=float(model_cfg.get("nll_b_init", -3.0)),
        nll_min_var=float(model_cfg.get("nll_min_var", 1.0e-6)),
        skip_sample=bool(model_cfg.get("skip_sample", False)),
        sample_type=str(model_cfg.get("sample_type", "linear")),
        sample_step=int(model_cfg.get("sample_step", 200)),
    ).to(device)


def _eval_group(
    *,
    label: str,
    zarr_path: Path,
    dataset: Dataset,
    model: BrownianBridge,
    device: torch.device,
    steps: torch.Tensor,
    direction: int,
    sampling_seeds: list[int],
    output_format: str,
    out_dir: Path,
    save_ids: bool,
    flush_every: int,
    meta_base: dict,
):
    group_name = _default_group_name(zarr_path)
    base_name = f"anomaly_score_{label}_dir{int(direction)}_{group_name}"
    out_meta = out_dir / f"{base_name}.json"
    tmp_meta = out_meta.with_suffix(out_meta.suffix + ".tmp")

    if output_format == "csv":
        out_data = out_dir / f"{base_name}.csv"
    elif output_format == "jsonl":
        out_data = out_dir / f"{base_name}.jsonl"
    elif output_format == "npz":
        out_data = out_dir / f"{base_name}.npz"
        tmp_data = out_data.with_name(out_data.stem + ".tmp.npz")
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    loader = DataLoader(
        dataset,
        batch_size=int(meta_base["batch_size"]),
        shuffle=False,
        num_workers=int(meta_base["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    total = len(dataset)
    print(f"\nInput [{label}]: {zarr_path}  samples={total}")
    cpu_states, cuda_states = _init_rng_states(sampling_seeds, device)

    def _write_meta(num_processed: int):
        meta = dict(meta_base)
        meta.update(
            {
                "input_zarr": str(zarr_path),
                "group": label,
                "direction": int(direction),
                "num_samples": int(total),
                "num_processed": int(min(num_processed, total)),
            }
        )
        tmp_meta.write_text(json.dumps(meta, indent=2))
        tmp_meta.replace(out_meta)

    if output_format == "npz":
        scores = np.full((total,), np.nan, dtype=np.float32)
        scores_norm = np.full((total,), np.nan, dtype=np.float32)
        ids: list[str] | None = [] if save_ids else None
        indices = np.arange(total, dtype=np.int64)

        def _write_checkpoint_npz(num_processed: int):
            payload = {
                "indices": indices,
                "anomaly_score": scores,
                "anomaly_score_norm": scores_norm,
                "sampling_seeds": np.asarray(sampling_seeds, dtype=np.int64),
            }
            if ids is not None and num_processed >= total:
                payload["ids"] = np.asarray(ids, dtype=np.str_)
            np.savez(tmp_data, **payload)
            tmp_data.replace(out_data)
            _write_meta(num_processed)

        offset = 0
        next_flush = flush_every if flush_every > 0 else None
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            names = batch.get("name")

            gt = target if direction == 1 else source
            cond = source if direction == 1 else target

            preds: list[torch.Tensor] = []
            for seed in sampling_seeds:
                _restore_rng_state(seed, cpu_states, cuda_states, device)

                if int(steps[0].item()) == 0:
                    x_t = gt
                else:
                    t = torch.full((gt.shape[0],), int(steps[0].item()), device=device, dtype=torch.long)
                    noise = torch.randn_like(gt)
                    x_t, _ = model.q_sample(gt, cond, t, noise=noise)

                pred = _denoise_from_xt(
                    model,
                    x_t,
                    cond,
                    steps,
                    source=source,
                    target=target,
                    direction=direction,
                )
                preds.append(pred.detach())
                _save_rng_state(seed, cpu_states, cuda_states, device)

            pred_stack = torch.stack(preds, dim=0)
            score, _ = _min_pixel_mse01(pred_stack, gt)
            score_norm = _min_pixel_mse01_norm(pred_stack, gt)
            scores[offset : offset + gt.shape[0]] = score.detach().cpu().numpy().astype(np.float32)
            scores_norm[offset : offset + gt.shape[0]] = score_norm.detach().cpu().numpy().astype(np.float32)

            if ids is not None and names is not None:
                ids.extend([str(n) for n in names])

            offset += gt.shape[0]
            if offset % 2048 == 0 or offset == total:
                print(f"  processed {offset}/{total}")
            if next_flush is not None and offset >= next_flush:
                _write_checkpoint_npz(offset)
                while next_flush is not None and offset >= next_flush:
                    next_flush += flush_every

        if offset != total:
            raise RuntimeError(f"Internal error: processed {offset} != total {total}")
        _write_checkpoint_npz(total)
        print(f"Saved: {out_data}")
        print(f"Saved: {out_meta}")
        return

    # CSV/JSONL streaming
    out_data.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    next_flush = flush_every if flush_every > 0 else None
    buffer_rows: list[list[object]] = []

    if output_format == "csv":
        handle = open(out_data, "w", newline="")
        writer = csv.writer(handle)
        if save_ids:
            writer.writerow(["index", "id", "anomaly_score", "anomaly_score_norm"])
        else:
            writer.writerow(["index", "anomaly_score", "anomaly_score_norm"])
    else:
        handle = open(out_data, "w")
        writer = None

    try:
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            names = batch.get("name")
            indices_batch = batch.get("index")

            gt = target if direction == 1 else source
            cond = source if direction == 1 else target

            preds: list[torch.Tensor] = []
            for seed in sampling_seeds:
                _restore_rng_state(seed, cpu_states, cuda_states, device)

                if int(steps[0].item()) == 0:
                    x_t = gt
                else:
                    t = torch.full((gt.shape[0],), int(steps[0].item()), device=device, dtype=torch.long)
                    noise = torch.randn_like(gt)
                    x_t, _ = model.q_sample(gt, cond, t, noise=noise)

                pred = _denoise_from_xt(
                    model,
                    x_t,
                    cond,
                    steps,
                    source=source,
                    target=target,
                    direction=direction,
                )
                preds.append(pred.detach())
                _save_rng_state(seed, cpu_states, cuda_states, device)

            pred_stack = torch.stack(preds, dim=0)
            score, _ = _min_pixel_mse01(pred_stack, gt)
            score_norm = _min_pixel_mse01_norm(pred_stack, gt)
            score = score.detach().cpu().numpy().astype(np.float32)
            score_norm = score_norm.detach().cpu().numpy().astype(np.float32)

            if names is None:
                name_list = [f"{zarr_path.name}:{offset + i}" for i in range(gt.shape[0])]
            else:
                name_list = [str(n) for n in names]

            for i in range(gt.shape[0]):
                idx = int(indices_batch[i]) if indices_batch is not None else (offset + i)
                val = float(score[i].item())
                val_norm = float(score_norm[i].item())
                if save_ids:
                    buffer_rows.append([idx, name_list[i], val, val_norm])
                else:
                    buffer_rows.append([idx, val, val_norm])

            offset += gt.shape[0]
            if offset % 2048 == 0 or offset == total:
                print(f"  processed {offset}/{total}")

            if next_flush is not None and offset >= next_flush:
                if output_format == "csv":
                    assert writer is not None
                    writer.writerows(buffer_rows)
                else:
                    for row in buffer_rows:
                        if save_ids:
                            obj = {
                                "index": int(row[0]),
                                "id": str(row[1]),
                                "anomaly_score": float(row[2]),
                                "anomaly_score_norm": float(row[3]),
                            }
                        else:
                            obj = {
                                "index": int(row[0]),
                                "anomaly_score": float(row[1]),
                                "anomaly_score_norm": float(row[2]),
                            }
                        handle.write(json.dumps(obj) + "\n")
                buffer_rows.clear()
                handle.flush()
                _write_meta(offset)
                while next_flush is not None and offset >= next_flush:
                    next_flush += flush_every

        if buffer_rows:
            if output_format == "csv":
                assert writer is not None
                writer.writerows(buffer_rows)
            else:
                for row in buffer_rows:
                    if save_ids:
                        obj = {
                            "index": int(row[0]),
                            "id": str(row[1]),
                            "anomaly_score": float(row[2]),
                            "anomaly_score_norm": float(row[3]),
                        }
                    else:
                        obj = {
                            "index": int(row[0]),
                            "anomaly_score": float(row[1]),
                            "anomaly_score_norm": float(row[2]),
                        }
                    handle.write(json.dumps(obj) + "\n")
            buffer_rows.clear()
            handle.flush()

        _write_meta(total)
    finally:
        handle.close()

    print(f"Saved: {out_data}")
    print(f"Saved: {out_meta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores using lens/nonlens zarrs.")
    parser.add_argument("--config", type=str, default="scirpts/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lens-zarr", type=str, action="append", required=True)
    parser.add_argument("--nonlens-zarr", type=str, action="append", required=True)
    parser.add_argument("--output-dir", type=str, default="metrics/anomaly_scores")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--direction", type=int, default=1, choices=[0, 1], help="1: source->target, 0: target->source")
    parser.add_argument("--start-t", type=int, default=200)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=2025)
    parser.add_argument("--sampling-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--samples-per-group", type=int, default=None, help="Randomly sample N per zarr (uses --index-file).")
    parser.add_argument("--index-file", type=str, default=None, help="NPZ path to save/load fixed indices.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument(
        "--output-format",
        type=str,
        default="csv",
        choices=["csv", "jsonl", "npz"],
        help="Output format (default: csv).",
    )
    parser.add_argument("--save-ids", action="store_true", help="Include string ids (names) in the output.")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help="Write partial results every N samples (default: 1000). Set <=0 to only write at the end.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = _resolve_path(repo_root, args.config) or args.config
    checkpoint_path = _resolve_path(repo_root, args.checkpoint) or args.checkpoint
    lens_zarrs = [_resolve_path(repo_root, p) or p for p in _as_list(args.lens_zarr)]
    nonlens_zarrs = [_resolve_path(repo_root, p) or p for p in _as_list(args.nonlens_zarr)]

    out_dir = Path(_resolve_path(repo_root, args.output_dir) or args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    cfg = yaml.safe_load(Path(config_path).read_text())
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    objective = str(model_cfg.get("objective", "eps")).lower()
    if objective not in {"eps", "grad", "maxnll"}:
        raise ValueError("model.objective must be one of: eps, grad, maxnll")

    source_key = data_cfg.get("source_key", "lsst")
    target_key = data_cfg.get("target_key", "euclid")
    euclid_to_rgb = bool(data_cfg.get("euclid_to_rgb", True))

    if args.sampling_seeds:
        sampling_seeds = [int(s) for s in args.sampling_seeds]
    else:
        sampling_seeds = [int(args.seed0) + i for i in range(int(args.num_candidates))]
    if len(sampling_seeds) < 1:
        raise ValueError("Need at least 1 candidate for anomaly score.")

    # Build model
    model = _build_shared_model(model_cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    steps = _build_truncated_steps(int(args.start_t), int(args.sample_steps), int(model.num_timesteps))
    steps = steps.to(device)

    index_file = Path(args.index_file).expanduser().resolve() if args.index_file else None

    meta_base = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "source_key": source_key,
        "target_key": target_key,
        "euclid_to_rgb": euclid_to_rgb,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "direction": int(args.direction),
        "start_t": int(args.start_t),
        "sample_steps": int(args.sample_steps),
        "num_candidates": int(args.num_candidates),
        "sampling_seeds": sampling_seeds,
        "output_format": str(args.output_format),
        "samples_per_group": None if args.samples_per_group is None else int(args.samples_per_group),
    }

    for zarr_path in lens_zarrs:
        zarr_path_p = Path(zarr_path)
        ds = ZarrPairDataset(
            zarr_path_p,
            source_key=source_key,
            target_key=target_key,
            euclid_to_rgb=euclid_to_rgb,
            max_samples=args.max_samples,
        )
        indices = None
        if args.samples_per_group:
            if index_file is None:
                raise ValueError("--samples-per-group requires --index-file")
            key = f"lens_{_default_group_name(zarr_path_p)}"
            indices = _get_fixed_indices(zarr_path_p, int(args.samples_per_group), index_file, key)
        ds_wrapped = _IndexWrapper(ds, indices)
        _eval_group(
            label="lens",
            zarr_path=zarr_path_p,
            dataset=ds_wrapped,
            model=model,
            device=device,
            steps=steps,
            direction=int(args.direction),
            sampling_seeds=sampling_seeds,
            output_format=str(args.output_format).lower(),
            out_dir=out_dir,
            save_ids=bool(args.save_ids),
            flush_every=int(args.flush_every),
            meta_base=meta_base,
        )

    for zarr_path in nonlens_zarrs:
        zarr_path_p = Path(zarr_path)
        ds = ZarrPairDataset(
            zarr_path_p,
            source_key=source_key,
            target_key=target_key,
            euclid_to_rgb=euclid_to_rgb,
            max_samples=args.max_samples,
        )
        indices = None
        if args.samples_per_group:
            if index_file is None:
                raise ValueError("--samples-per-group requires --index-file")
            key = f"nonlens_{_default_group_name(zarr_path_p)}"
            indices = _get_fixed_indices(zarr_path_p, int(args.samples_per_group), index_file, key)
        ds_wrapped = _IndexWrapper(ds, indices)
        _eval_group(
            label="nonlens",
            zarr_path=zarr_path_p,
            dataset=ds_wrapped,
            model=model,
            device=device,
            steps=steps,
            direction=int(args.direction),
            sampling_seeds=sampling_seeds,
            output_format=str(args.output_format).lower(),
            out_dir=out_dir,
            save_ids=bool(args.save_ids),
            flush_every=int(args.flush_every),
            meta_base=meta_base,
        )


if __name__ == "__main__":
    main()
