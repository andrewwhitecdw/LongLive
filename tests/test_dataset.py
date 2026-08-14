import tempfile
from pathlib import Path

import pytest

from utils.dataset import MultiTextConcatDataset


def test_multitext_concat_dataset_rejects_non_positive_num_blocks():
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_path = Path(tmpdir) / "prompts.txt"
        prompt_path.write_text("a cat jumping\na dog running\n")

        with pytest.raises(ValueError, match="num_blocks must be a positive integer"):
            MultiTextConcatDataset(str(prompt_path), num_blocks=0)

        with pytest.raises(ValueError, match="num_blocks must be a positive integer"):
            MultiTextConcatDataset(str(prompt_path), num_blocks=-1)
