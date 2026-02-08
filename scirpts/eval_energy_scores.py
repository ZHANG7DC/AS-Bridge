from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
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


def _build_linear_steps(num_timesteps: int, num_steps: int | None) -> torch.Tensor | None:
    if num_steps is None:
        return None
    num_steps = int(num_steps)
    if num_steps <= 0:
        return None

    num_timesteps = int(num_timesteps)
    num_steps = min(num_steps, num_timesteps)
    if num_steps == 1:
        return torch.tensor([0], dtype=torch.long)

    raw = torch.linspace(num_timesteps - 1, 0, num_steps).round().long()
    raw[0] = num_timesteps - 1
    raw[-1] = 0

    steps: list[int] = []
    last = None
    for s in raw.tolist():
        s = int(s)
        if last is None or s < last:
            steps.append(s)
            last = s
    if steps[-1] != 0:
        steps.append(0)
    return torch.tensor(steps, dtype=torch.long)


@torch.no_grad()
def _sample_with_steps(model: BrownianBridge, steps: torch.Tensor | None, **kwargs) -> torch.Tensor:
    if steps is None:
        return model.sample(**kwargs)
    old = model.steps
    model.steps = steps
    try:
        return model.sample(**kwargs)
    finally:
        model.steps = old


def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def _energy_score01(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Empirical Energy Score (per example).

    pred: (K,B,C,H,W) in [-1,1]
    gt:   (B,C,H,W)   in [-1,1]
    returns: (B,) in [0,1] space (distance units).
    """

    pred01 = _to01(pred)
    gt01 = _to01(gt)
    K, B = int(pred01.shape[0]), int(pred01.shape[1])
    pred_flat = pred01.reshape(K, B, -1)
    gt_flat = gt01.reshape(B, -1)

    dist_to_gt = torch.linalg.vector_norm(pred_flat - gt_flat.unsqueeze(0), dim=-1).mean(dim=0)
    diffs = pred_flat[:, None, :, :] - pred_flat[None, :, :, :]
    dist_pair = torch.linalg.vector_norm(diffs, dim=-1).mean(dim=(0, 1))

    return dist_to_gt - 0.5 * dist_pair


def _init_rng_states(seeds: list[int], device: torch.device):
    """Create per-seed RNG states so we can interleave seeds per batch deterministically."""

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
        skip_sample=bool(model_cfg.get("skip_sample", False)),
        sample_type=str(model_cfg.get("sample_type", "linear")),
        sample_step=int(model_cfg.get("sample_step", 200)),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-sample energy score.")
    parser.add_argument("--config", type=str, default="scirpts/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-zarr", type=str, action="append", required=True)
    parser.add_argument("--output-dir", type=str, default="metrics/energy_scores")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-sampling-steps", type=int, default=200)
    parser.add_argument("--direction", type=int, default=1, choices=[0, 1], help="1: source->target, 0: target->source")
    parser.add_argument("--num-generations", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=2025)
    parser.add_argument("--sampling-seeds", type=int, nargs="*", default=None)
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
    checkpoint_path = _resolve_path(repo_root, args.checkpoint)
    input_zarrs = [_resolve_path(repo_root, p) or p for p in _as_list(args.input_zarr)]

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

    if not checkpoint_path:
        raise ValueError("Provide --checkpoint.")

    if args.sampling_seeds:
        sampling_seeds = [int(s) for s in args.sampling_seeds]
    else:
        sampling_seeds = [int(args.seed0) + i for i in range(int(args.num_generations))]
    if len(sampling_seeds) < 2:
        raise ValueError("Energy score requires an ensemble; provide at least 2 seeds.")

    model = _build_shared_model(model_cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    print(f"[init] Loaded checkpoint: {checkpoint_path}")
    model.eval()
    steps = _build_linear_steps(int(model.num_timesteps), int(args.num_sampling_steps)) if args.num_sampling_steps else None

    for zarr_path in input_zarrs:
        zarr_path_p = Path(zarr_path)
        group_name = _default_group_name(zarr_path_p)
        mode_tag = "shared"
        base_name = f"energy_score_{mode_tag}_dir{int(args.direction)}_{group_name}"
        out_meta = out_dir / f"{base_name}.json"
        tmp_meta = out_meta.with_suffix(out_meta.suffix + ".tmp")

        output_format = str(args.output_format).lower()
        if output_format == "csv":
            out_data = out_dir / f"{base_name}.csv"
        elif output_format == "jsonl":
            out_data = out_dir / f"{base_name}.jsonl"
        elif output_format == "npz":
            out_data = out_dir / f"{base_name}.npz"
            # NOTE: numpy.savez appends ".npz" if the filename doesn't end with ".npz".
            tmp_data = out_data.with_name(out_data.stem + ".tmp.npz")
        else:
            raise ValueError(f"Unknown output format: {output_format}")

        ds = ZarrPairDataset(
            zarr_path_p,
            source_key=source_key,
            target_key=target_key,
            euclid_to_rgb=euclid_to_rgb,
            max_samples=args.max_samples,
        )
        loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        total = len(ds)
        print(f"\nInput: {zarr_path_p}  samples={total}")
        cpu_states, cuda_states = _init_rng_states(sampling_seeds, device)

        def _write_meta(num_processed: int):
            meta = {
                "config": str(config_path),
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "input_zarr": str(zarr_path_p),
                "model_mode": mode_tag,
                "direction": int(args.direction),
                "source_key": source_key,
                "target_key": target_key,
                "euclid_to_rgb": euclid_to_rgb,
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "num_sampling_steps": None if args.num_sampling_steps is None else int(args.num_sampling_steps),
                "sampling_seeds": sampling_seeds,
                "output_format": output_format,
                "num_samples": int(total),
                "num_processed": int(min(num_processed, total)),
            }
            tmp_meta.write_text(json.dumps(meta, indent=2))
            tmp_meta.replace(out_meta)

        if output_format == "npz":
            energy_scores = np.full((total,), np.nan, dtype=np.float32)
            ids: list[str] | None = [] if args.save_ids else None
            indices = np.arange(total, dtype=np.int64)

            def _write_checkpoint_npz(num_processed: int):
                payload = {
                    "indices": indices,
                    "energy_score": energy_scores,
                    "sampling_seeds": np.asarray(sampling_seeds, dtype=np.int64),
                }
                if ids is not None and num_processed >= total:
                    payload["ids"] = np.asarray(ids, dtype=np.str_)
                np.savez(tmp_data, **payload)
                tmp_data.replace(out_data)
                _write_meta(num_processed)

            offset = 0
            flush_every = int(args.flush_every)
            next_flush = flush_every if flush_every > 0 else None
            for batch in loader:
                source = batch["source"].to(device)
                target = batch["target"].to(device)
                names = batch.get("name")

                B = int(source.shape[0])
                gt = target if args.direction == 1 else source

                preds: list[torch.Tensor] = []
                for seed in sampling_seeds:
                    _restore_rng_state(seed, cpu_states, cuda_states, device)

                    if args.direction == 1:
                        zeros_tgt = torch.zeros_like(target)
                        pred = _sample_with_steps(model, steps, source=source, target=zeros_tgt, direction=1)
                    else:
                        zeros_src = torch.zeros_like(source)
                        pred = _sample_with_steps(model, steps, source=zeros_src, target=target, direction=0)

                    preds.append(pred.detach())
                    _save_rng_state(seed, cpu_states, cuda_states, device)

                pred_stack = torch.stack(preds, dim=0)  # (K,B,C,H,W) on device
                es = _energy_score01(pred_stack, gt)  # (B,)
                energy_scores[offset : offset + B] = es.detach().cpu().numpy().astype(np.float32)

                if ids is not None and names is not None:
                    ids.extend([str(n) for n in names])

                offset += B
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
            continue

        # Streaming CSV / JSONL
        out_data.parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        flush_every = int(args.flush_every)
        next_flush = flush_every if flush_every > 0 else None
        buffer_rows: list[list[object]] = []

        running_sum = 0.0
        running_count = 0

        if output_format == "csv":
            handle = open(out_data, "w", newline="")
            writer = csv.writer(handle)
            if args.save_ids:
                writer.writerow(["index", "id", "energy_score"])
            else:
                writer.writerow(["index", "energy_score"])
        else:
            handle = open(out_data, "w")
            writer = None

        try:
            for batch in loader:
                source = batch["source"].to(device)
                target = batch["target"].to(device)
                names = batch.get("name")

                B = int(source.shape[0])
                gt = target if args.direction == 1 else source

                preds: list[torch.Tensor] = []
                for seed in sampling_seeds:
                    _restore_rng_state(seed, cpu_states, cuda_states, device)

                    if args.direction == 1:
                        zeros_tgt = torch.zeros_like(target)
                        pred = _sample_with_steps(model, steps, source=source, target=zeros_tgt, direction=1)
                    else:
                        zeros_src = torch.zeros_like(source)
                        pred = _sample_with_steps(model, steps, source=zeros_src, target=target, direction=0)

                    preds.append(pred.detach())
                    _save_rng_state(seed, cpu_states, cuda_states, device)

                pred_stack = torch.stack(preds, dim=0)  # (K,B,C,H,W) on device
                es = _energy_score01(pred_stack, gt).detach().cpu().numpy().astype(np.float32)  # (B,)

                if names is None:
                    name_list = [f"{zarr_path_p.name}:{offset + i}" for i in range(B)]
                else:
                    name_list = [str(n) for n in names]

                for i in range(B):
                    idx = offset + i
                    score = float(es[i].item())
                    if args.save_ids:
                        buffer_rows.append([idx, name_list[i], score])
                    else:
                        buffer_rows.append([idx, score])
                    running_sum += score
                    running_count += 1

                offset += B
                if offset % 2048 == 0 or offset == total:
                    print(f"  processed {offset}/{total}")

                if next_flush is not None and offset >= next_flush:
                    if output_format == "csv":
                        assert writer is not None
                        writer.writerows(buffer_rows)
                    else:
                        for row in buffer_rows:
                            if args.save_ids:
                                obj = {"index": int(row[0]), "id": str(row[1]), "energy_score": float(row[2])}
                            else:
                                obj = {"index": int(row[0]), "energy_score": float(row[1])}
                            handle.write(json.dumps(obj) + "\n")
                    buffer_rows.clear()
                    handle.flush()
                    _write_meta(offset)
                    while next_flush is not None and offset >= next_flush:
                        next_flush += flush_every

            if offset != total:
                raise RuntimeError(f"Internal error: processed {offset} != total {total}")

            if buffer_rows:
                if output_format == "csv":
                    assert writer is not None
                    writer.writerows(buffer_rows)
                else:
                    for row in buffer_rows:
                        if args.save_ids:
                            obj = {"index": int(row[0]), "id": str(row[1]), "energy_score": float(row[2])}
                        else:
                            obj = {"index": int(row[0]), "energy_score": float(row[1])}
                        handle.write(json.dumps(obj) + "\n")
                buffer_rows.clear()
                handle.flush()

            _write_meta(total)
        finally:
            handle.close()

        mean_es = running_sum / max(1, running_count)
        print(f"Saved: {out_data}")
        print(f"Saved: {out_meta}")
        print(f"Mean EnergyScore01: {mean_es:.6f}")


if __name__ == "__main__":
    main()
