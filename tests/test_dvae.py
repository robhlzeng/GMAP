import torch
import pytest
from gmap.models.dvae import DVAE

@pytest.fixture
def dvae():
    return DVAE(
        group_size=32,
        encoder_dims=[64, 128, 256],
        codebook_size=8192,
        codebook_dim=256,
    )

def test_dvae_output_shape(dvae):
    patches = torch.randn(2, 512, 32, 3)
    logits, recon = dvae(patches, temperature=1.0)
    assert logits.shape == (2, 512, 8192)
    assert recon.shape == (2, 512, 32, 3)

def test_dvae_get_tokens(dvae):
    patches = torch.randn(2, 512, 32, 3)
    tokens = dvae.get_tokens(patches)
    assert tokens.shape == (2, 512)
    assert tokens.dtype == torch.long
    assert tokens.max() < 8192

def test_dvae_codebook_lookup(dvae):
    tokens = torch.randint(0, 8192, (2, 512))
    embeddings = dvae.codebook_lookup(tokens)
    assert embeddings.shape == (2, 512, 256)
