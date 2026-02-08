from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

try:
    import zarr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("zarr is required. Install with: python -m pip install zarr") from exc


def _normalize(arr):
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 127.5 - 1.0
    arr = arr.astype(np.float32, copy=False)
    if arr.min() >= 0.0 and arr.max() <= 1.0:
        return arr * 2.0 - 1.0
    return arr


def _to_chw(arr, *, to_rgb=False):
    """Convert HxW / HxWxC / CxHxW to CxHxW.

    If to_rgb is True and input is 2D, repeat to 3 channels.
    Channel order is auto-detected for 3D inputs.
    """
    if arr.ndim == 2:
        if to_rgb:
            return np.repeat(arr[None, :, :], 3, axis=0)
        return arr[None, :, :]
    if arr.ndim == 3:
        # If channel-last (HxWxC), transpose to CxHxW.
        if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
            return np.transpose(arr, (2, 0, 1))
        return arr
    raise ValueError(f"Unsupported array ndim={arr.ndim} for conversion to CHW.")


class ZarrPairDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        source_key="lsst",
        target_key="euclid",
        euclid_to_rgb=True,
        max_samples=None,
    ):
        self.zarr_path = Path(zarr_path)
        self.group = zarr.open_group(str(self.zarr_path), mode="r")
        self.source = self.group[source_key]
        self.target = self.group[target_key]
        self.ids = self.group.get("id", None)
        self.euclid_to_rgb = bool(euclid_to_rgb)
        self.max_samples = int(max_samples) if max_samples else None

        if self.source.shape[0] != self.target.shape[0]:
            raise ValueError("Source and target must have the same number of samples.")

    def __len__(self):
        total = int(self.source.shape[0])
        if self.max_samples is None:
            return total
        return min(self.max_samples, total)

    def __getitem__(self, idx):
        idx = int(idx)
        src = _normalize(self.source[idx])
        tgt = _normalize(self.target[idx])

        src = _to_chw(src, to_rgb=False)
        tgt = _to_chw(tgt, to_rgb=self.euclid_to_rgb)

        name = f"{self.zarr_path.name}:{idx}"
        if self.ids is not None:
            name = self.ids[idx]
            if hasattr(name, "item"):
                name = name.item()
            if isinstance(name, bytes):
                name = name.decode("utf-8")

        return {
            "source": torch.from_numpy(src).float(),
            "target": torch.from_numpy(tgt).float(),
            "name": str(name),
        }


