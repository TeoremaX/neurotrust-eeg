import torch

from neurotrust_eeg.models import TinyTCN, TinyTCNWithPrototypeMemory


def test_tiny_tcn_output_shape():
    model = TinyTCN(in_channels=18)
    x = torch.randn(4, 18, 1024)

    y = model(x)

    assert y.shape == (4,)


def test_prototype_memory_model_output_shape():
    model = TinyTCNWithPrototypeMemory(in_channels=18)
    x = torch.randn(4, 18, 1024)

    y = model(x)

    assert y.shape == (4,)