import torch
import pytest
from gmap.models.msfe import MSFE
from gmap.models.transformer import TransformerEncoder

def test_transformer_encoder():
    enc = TransformerEncoder(dim=384, depth=2, heads=6, mlp_ratio=4.0)
    x = torch.randn(2, 64, 384)
    out = enc(x)
    assert out.shape == (2, 64, 384)

def test_msfe_output_shapes():
    msfe = MSFE(
        n_points=8192,
        scales=[(512, 32), (256, 8), (64, 8)],
        embed_dim=384,
        depth=6,
        heads=6,
    )
    xyz = torch.randn(2, 8192, 3)
    features, centers = msfe(xyz)
    assert len(features) == 3
    assert features[0].shape == (2, 512, 384)
    assert features[1].shape == (2, 256, 384)
    assert features[2].shape == (2, 64, 384)
    assert centers[0].shape == (2, 512, 3)
