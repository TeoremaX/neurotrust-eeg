from pathlib import Path

import torch

from neurotrust_eeg.channels import get_default_channels
from neurotrust_eeg.config import deep_update, load_config
from neurotrust_eeg.models import TinyTCN
from neurotrust_eeg.utils import count_trainable_parameters, ensure_dir, get_device


def test_deep_update_nested_dictionary():
    base = {
        "model": {
            "hidden_channels": 32,
            "dropout": 0.2,
        },
        "seed": 42,
    }

    update = {
        "model": {
            "dropout": 0.5,
        }
    }

    result = deep_update(base, update)

    assert result["seed"] == 42
    assert result["model"]["hidden_channels"] == 32
    assert result["model"]["dropout"] == 0.5


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 123
model:
  name: tiny_tcn
  hidden_channels: 16
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["seed"] == 123
    assert config["model"]["name"] == "tiny_tcn"
    assert config["model"]["hidden_channels"] == 16


def test_default_channels_are_unique_and_expected_length():
    channels = get_default_channels()

    assert len(channels) == 18
    assert len(set(channels)) == 18
    assert "FP1-F7" in channels
    assert "CZ-PZ" in channels


def test_utils_create_directory_and_count_parameters(tmp_path):
    output_dir = ensure_dir(tmp_path / "outputs")

    assert Path(output_dir).exists()

    model = TinyTCN(in_channels=18)
    parameter_count = count_trainable_parameters(model)

    assert parameter_count > 0


def test_get_device_returns_torch_device():
    device = get_device()

    assert isinstance(device, torch.device)