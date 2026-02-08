import argparse
import os
import sys

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
import yaml

from dataset import BalancedZarrPairDataset, ZarrPairDataset
from model import BrownianBridge
from unet import UNet
from utils import seed_everything

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from PIL import Image

    def _save_image(path, array):
        Image.fromarray(array).save(path)

except Exception:  # pragma: no cover
    import imageio.v2 as imageio

    def _save_image(path, array):
        imageio.imwrite(path, array)


def _to_uint8(arr):
    arr = (arr + 1.0) * 127.5
    arr = arr.clamp(0, 255).byte()
    return arr


def _make_grid(inputs, targets, outputs):
    rows = []
    for i in range(inputs.shape[0]):
        row = torch.cat([inputs[i], targets[i], outputs[i]], dim=2)
        rows.append(row)
    grid = torch.cat(rows, dim=1)
    return grid


def _build_linear_steps(num_timesteps, num_steps):
    num_steps = int(num_steps)
    if num_steps <= 0:
        return None
    if num_steps == 1:
        return torch.tensor([0], dtype=torch.long)
    steps = torch.linspace(num_timesteps - 1, 0, num_steps).round().long()
    steps[0] = num_timesteps - 1
    steps[-1] = 0
    return steps


@torch.no_grad()
def _sample_with_steps(model, steps, **kwargs):
    if steps is None:
        return model.sample(**kwargs)
    original_steps = model.steps
    model.steps = steps
    try:
        return model.sample(**kwargs)
    finally:
        model.steps = original_steps


def _visualize_samples(model, batch, device, out_dir, step):
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        source_px = batch["source"].to(device)
        target_px = batch["target"].to(device)
        model.eval()
        pred_xy = model.sample(source=source_px, target=target_px, direction=1)
        pred_yx = model.sample(source=source_px, target=target_px, direction=0)

    vis_xy = _make_grid(source_px, target_px, pred_xy)
    vis_xy = _to_uint8(vis_xy).permute(1, 2, 0).cpu().numpy()
    _save_image(os.path.join(out_dir, f"sample_xy_{step:06d}.png"), vis_xy)

    vis_yx = _make_grid(target_px, source_px, pred_yx)
    vis_yx = _to_uint8(vis_yx).permute(1, 2, 0).cpu().numpy()
    _save_image(os.path.join(out_dir, f"sample_yx_{step:06d}.png"), vis_yx)

    model.train()


def _load_config(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        if str(value).lower() == "none":
            return []
        return [value]
    return [v for v in value if v and str(v).lower() != "none"]


def _limit_dataset(dataset, max_samples):
    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0:
        return Subset(dataset, [])
    if max_samples >= len(dataset):
        return dataset
    return Subset(dataset, range(max_samples))


def _build_concat_pair_dataset(paths, source_key, target_key, euclid_to_rgb):
    paths = _as_list(paths)
    if not paths:
        return None
    datasets = [
        ZarrPairDataset(
            path,
            source_key=source_key,
            target_key=target_key,
            euclid_to_rgb=euclid_to_rgb,
        )
        for path in paths
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _build_train_dataset(data_cfg):
    source_key = data_cfg.get("source_key", "lsst")
    target_key = data_cfg.get("target_key", "euclid")
    euclid_to_rgb = bool(data_cfg.get("euclid_to_rgb", True))
    max_samples = data_cfg.get("max_samples")

    balance = bool(data_cfg.get("balance", False))
    lens_paths = _as_list(data_cfg.get("lens_path"))
    nonlens_paths = _as_list(data_cfg.get("nonlens_path"))
    zarr_paths = _as_list(data_cfg.get("zarr_path"))

    if balance and lens_paths and nonlens_paths:
        return BalancedZarrPairDataset(
            lens_path=lens_paths,
            nonlens_path=nonlens_paths,
            source_key=source_key,
            target_key=target_key,
            euclid_to_rgb=euclid_to_rgb,
            max_samples=max_samples,
        )

    dataset = _build_concat_pair_dataset(zarr_paths, source_key, target_key, euclid_to_rgb)
    if dataset is None:
        parts = []
        if lens_paths:
            parts.append(_build_concat_pair_dataset(lens_paths, source_key, target_key, euclid_to_rgb))
        if nonlens_paths:
            parts.append(_build_concat_pair_dataset(nonlens_paths, source_key, target_key, euclid_to_rgb))
        if not parts:
            raise ValueError("Provide data.zarr_path or data.lens_path/nonlens_path for training.")
        dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return _limit_dataset(dataset, max_samples)


def _build_cycle_dataset(data_cfg):
    source_key = data_cfg.get("source_key", "lsst")
    target_key = data_cfg.get("target_key", "euclid")
    euclid_to_rgb = bool(data_cfg.get("euclid_to_rgb", True))
    max_samples = data_cfg.get("cycle_max_samples")

    balance = bool(data_cfg.get("balance", False))
    lens_paths = _as_list(data_cfg.get("lens_path_cycle"))
    nonlens_paths = _as_list(data_cfg.get("nonlens_path_cycle") or data_cfg.get("nonLens_path_cycle"))
    zarr_paths = _as_list(data_cfg.get("zarr_path_cycle"))

    if balance and lens_paths and nonlens_paths:
        return BalancedZarrPairDataset(
            lens_path=lens_paths,
            nonlens_path=nonlens_paths,
            source_key=source_key,
            target_key=target_key,
            euclid_to_rgb=euclid_to_rgb,
            max_samples=max_samples,
        )

    dataset = _build_concat_pair_dataset(zarr_paths, source_key, target_key, euclid_to_rgb)
    if dataset is not None:
        return _limit_dataset(dataset, max_samples)

    parts = []
    if lens_paths:
        parts.append(_build_concat_pair_dataset(lens_paths, source_key, target_key, euclid_to_rgb))
    if nonlens_paths:
        parts.append(_build_concat_pair_dataset(nonlens_paths, source_key, target_key, euclid_to_rgb))
    if not parts:
        return None
    dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return _limit_dataset(dataset, max_samples)


def main():
    parser = argparse.ArgumentParser(description="Train Brownian Bridge diffusion (LSST -> Euclid).")
    parser.add_argument("--config", type=str, default="scirpts/config.yaml")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to override train.resume_path from config.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.resume:
        config.setdefault("train", {})["resume_path"] = args.resume
    print(config)

    seed_everything(config["seed"])
    device = torch.device(config["device"])

    data_cfg = config["data"]
    dataset = _build_train_dataset(data_cfg)

    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )

    model_cfg = config["model"]
    objective = str(model_cfg.get("objective", "eps")).lower()
    if objective not in {"eps", "grad", "maxnll"}:
        raise ValueError("model.objective must be one of: eps, grad, maxnll")

    channels = int(model_cfg.get("channels", 3))

    bridge_kwargs = dict(
        num_timesteps=model_cfg.get("num_timesteps", 1000),
        mt_type=model_cfg.get("mt_type", "linear"),
        max_var=model_cfg.get("max_var", 1.0),
        eta=model_cfg.get("eta", 1.0),
        objective=objective,
        loss_type=model_cfg.get("loss_type", "mse"),
        channels=channels,
        skip_sample=model_cfg.get("skip_sample", False),
        sample_type=model_cfg.get("sample_type", "linear"),
        sample_step=model_cfg.get("sample_step", 200),
    )

    net = UNet(
        in_channels=channels * 3,
        out_channels=channels * 2,
        base_channels=model_cfg.get("base_channels", 64),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        time_dim=model_cfg.get("time_dim", 256),
    )
    model = BrownianBridge(denoise_fn=net, **bridge_kwargs).to(device)
    optim_params = model.parameters()

    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    start_epoch = 1
    resume_path = train_cfg.get("resume_path")
    if resume_path and resume_path != "none":
        checkpoint = torch.load(resume_path, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint.get("optimizer", optimizer.state_dict()))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed from: {resume_path}")

    save_path = train_cfg.get("save_path", "AS-Bridge/ckpts/eps")
    os.makedirs(save_path, exist_ok=True)

    sample_every = int(train_cfg.get("sample_every", 0) or 0)
    sample_dir = train_cfg.get("sample_dir", "AS-Bridge/samples/eps")
    sample_count = int(train_cfg.get("sample_count", 4))
    cycle_weight = float(train_cfg.get("cycle_weight", 0.0))
    cycle_sample_steps = int(train_cfg.get("cycle_sample_steps", 20) or 0)
    cycle_steps = None
    test_loader = None
    test_sample_dir = None
    lens_path_test = _as_list(data_cfg.get("lens_path_test"))
    nonlens_path_test = _as_list(data_cfg.get("nonlens_path_test"))
    if lens_path_test or nonlens_path_test:
        if data_cfg.get("balance", False) and lens_path_test and nonlens_path_test:
            test_dataset = BalancedZarrPairDataset(
                lens_path=lens_path_test,
                nonlens_path=nonlens_path_test,
                source_key=data_cfg.get("source_key", "lsst"),
                target_key=data_cfg.get("target_key", "euclid"),
                euclid_to_rgb=bool(data_cfg.get("euclid_to_rgb", True)),
            )
        else:
            parts = []
            if lens_path_test:
                parts.append(
                    _build_concat_pair_dataset(
                        lens_path_test,
                        data_cfg.get("source_key", "lsst"),
                        data_cfg.get("target_key", "euclid"),
                        bool(data_cfg.get("euclid_to_rgb", True)),
                    )
                )
            if nonlens_path_test:
                parts.append(
                    _build_concat_pair_dataset(
                        nonlens_path_test,
                        data_cfg.get("source_key", "lsst"),
                        data_cfg.get("target_key", "euclid"),
                        bool(data_cfg.get("euclid_to_rgb", True)),
                    )
                )
            test_dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        test_loader = DataLoader(
            test_dataset,
            batch_size=sample_count,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        test_sample_dir = os.path.join(sample_dir, "test")

    cycle_loader = None
    cycle_iter = None
    if cycle_weight > 0.0:
        cycle_dataset = _build_cycle_dataset(data_cfg)
        if cycle_dataset is None:
            print("Info: cycle_weight>0 but no cycle paths provided; disabling cycle consistency.")
            cycle_weight = 0.0
        else:
            cycle_loader = DataLoader(
                cycle_dataset,
                batch_size=int(data_cfg.get("cycle_batch_size", data_cfg["batch_size"])),
                shuffle=True,
                num_workers=int(data_cfg.get("cycle_num_workers", data_cfg.get("num_workers", 0))),
                pin_memory=True,
                drop_last=True,
            )
            cycle_iter = iter(cycle_loader)
            if cycle_sample_steps > 0:
                cycle_steps = _build_linear_steps(model.num_timesteps, cycle_sample_steps)

    num_epochs = int(train_cfg.get("num_epochs", 1))
    pbar = tqdm(range(start_epoch, num_epochs + 1), dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        running = 0.0
        batch_iter = loader
        if train_cfg.get("batch_progress", True):
            batch_iter = tqdm(loader, dynamic_ncols=True, leave=False)
        reverse_weight = float(train_cfg.get("reverse_weight", 1.0))
        for batch in batch_iter:
            target = batch["target"].to(device)
            source = batch["source"].to(device)

            optimizer.zero_grad()
            loss_xy, _ = model(
                target,
                source,
                source=source,
                target=target,
                direction=1,
            )
            loss_yx, _ = model(
                source,
                target,
                source=source,
                target=target,
                direction=0,
            )
            loss = loss_xy + reverse_weight * loss_yx

            loss_cycle_source = None
            loss_cycle_target = None
            if cycle_weight > 0.0 and cycle_loader is not None:
                try:
                    cycle_batch = next(cycle_iter)
                except StopIteration:
                    cycle_iter = iter(cycle_loader)
                    cycle_batch = next(cycle_iter)
                cycle_target = cycle_batch["target"].to(device)
                cycle_source = cycle_batch["source"].to(device)
                with torch.no_grad():
                    zeros_source = torch.zeros_like(cycle_source)
                    zeros_target = torch.zeros_like(cycle_target)
                    pred_target = _sample_with_steps(
                        model,
                        cycle_steps,
                        source=cycle_source,
                        target=zeros_target,
                        direction=1,
                    )
                    pred_source = _sample_with_steps(
                        model,
                        cycle_steps,
                        source=zeros_source,
                        target=cycle_target,
                        direction=0,
                    )
                loss_cycle_source, _ = model(
                    cycle_source,
                    pred_target,
                    source=cycle_source,
                    target=pred_target,
                    direction=0,
                )
                loss_cycle_target, _ = model(
                    cycle_target,
                    pred_source,
                    source=pred_source,
                    target=cycle_target,
                    direction=1,
                )
                loss = loss + cycle_weight * (loss_cycle_source + loss_cycle_target)
            loss.backward()
            optimizer.step()

            running += loss.item()
            if train_cfg.get("batch_progress", True):
                postfix = {"loss": loss.item(), "loss_xy": loss_xy.item(), "loss_yx": loss_yx.item()}
                if cycle_weight > 0.0 and loss_cycle_source is not None:
                    postfix["cycle_src"] = loss_cycle_source.item()
                    postfix["cycle_tgt"] = loss_cycle_target.item()
                batch_iter.set_postfix(postfix)

        avg_loss = running / max(1, len(loader))
        pbar.set_description(f"loss: {avg_loss:.6f}")

        if epoch % int(train_cfg.get("save_every", 1)) == 0:
            ckpt_path = os.path.join(save_path, f"bbdm_epoch_{epoch}.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

        if sample_every and epoch % sample_every == 0:
            batch = next(iter(loader))
            if sample_count < batch["source"].shape[0]:
                for key in ("source", "target"):
                    batch[key] = batch[key][:sample_count]
            _visualize_samples(
                model,
                batch,
                device,
                sample_dir,
                epoch,
            )
            if test_loader is not None:
                test_batch = next(iter(test_loader))
                if sample_count < test_batch["source"].shape[0]:
                    for key in ("source", "target"):
                        test_batch[key] = test_batch[key][:sample_count]
                _visualize_samples(
                    model,
                    test_batch,
                    device,
                    test_sample_dir,
                    epoch,
                )


if __name__ == "__main__":
    main()
