import pytest
from pathlib import Path
from utils.dataset import MultiVideoConcatDataset


def test_multi_video_concat_dataset_folders_are_sorted(tmp_path):
    root = tmp_path / "dataset"
    video_dir = root / "video"
    caption_dir = root / "caption"

    for name in ("zzz", "aaa", "mmm"):
        (video_dir / name).mkdir(parents=True)
    caption_dir.mkdir(parents=True)

    ds = MultiVideoConcatDataset(
        str(root),
        video_size=(64, 64),
        total_frames=29,
    )

    names = [f.name for f in ds.folders]
    assert names == ["aaa", "mmm", "zzz"]
    assert names == sorted(names)
