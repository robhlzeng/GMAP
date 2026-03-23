import torch
import numpy as np
import pytest
from gmap.eval.metrics import compute_miou, compute_axis_error, compute_position_error

def test_compute_miou():
    pred = torch.tensor([0, 0, 1, 1, 2, 2])
    target = torch.tensor([0, 0, 1, 2, 2, 2])
    miou = compute_miou(pred, target, n_classes=3)
    assert 0 <= miou <= 1

def test_compute_axis_error():
    pred = torch.tensor([[1.0, 0, 0]])
    target = torch.tensor([[0.0, 1.0, 0]])
    error = compute_axis_error(pred, target)
    assert abs(error - 90.0) < 1.0

def test_compute_position_error():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 2.0, 3.1]])
    error = compute_position_error(pred, target)
    assert error < 1.0
