import torch

from pytorchlstm.model import LSTMModel


def test_output_shape():
    model = LSTMModel(hidden=8)
    x = torch.zeros(4, 10, 1)  # batch=4, time=10, features=1
    assert model(x).shape == (4,)


def test_output_has_no_nan():
    model = LSTMModel(hidden=8)
    out = model(torch.randn(4, 10, 1))
    assert not torch.isnan(out).any()


def test_multivariate_input():
    model = LSTMModel(n_features=3, hidden=8)
    assert model(torch.randn(4, 10, 3)).shape == (4,)
