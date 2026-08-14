import pathlib

import pytest
import yaml


def test_te_config_uses_te_fallback():
    cfg_path = pathlib.Path("configs/nvfp4/inference_i2v_nvfp4.yaml")
    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text())
    # When TransformerEngine is selected and a TE checkpoint is used, the
    # FourOverSix fallback flag must be set to true (i.e. use the TE path).
    assert cfg["model_quant_use_transformer_engine"] is True
    assert cfg["model_quant_te_fallback_to_fouroversix"] is True
