import torch
import torch.nn as nn
import torch.nn.functional as F

class DVAEEncoder(nn.Module):
    """Mini-PointNet encoder for each patch."""

    def __init__(self, in_dim: int = 3, dims: list[int] = [64, 128, 256]):
        super().__init__()
        layers = []
        prev = in_dim
        for d in dims:
            layers.append(nn.Conv1d(prev, d, 1))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU(inplace=True))
            prev = d
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*M, 3, K) -> (B*M, C)"""
        feat = self.mlp(x)
        feat = feat.max(dim=-1)[0]
        return feat


class DVAEDecoder(nn.Module):
    """Decode codebook embeddings back to point patches."""

    def __init__(self, codebook_dim: int, group_size: int, dims: list[int] = [256, 128]):
        super().__init__()
        layers = []
        prev = codebook_dim
        for d in dims:
            layers.append(nn.Linear(prev, d))
            layers.append(nn.ReLU(inplace=True))
            prev = d
        layers.append(nn.Linear(prev, group_size * 3))
        self.mlp = nn.Sequential(*layers)
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*M, codebook_dim) -> (B*M, K, 3)"""
        out = self.mlp(x)
        return out.view(-1, self.group_size, 3)


class DVAE(nn.Module):
    """Discrete VAE tokenizer for point cloud patches (Point-BERT style)."""

    def __init__(
        self,
        group_size: int = 32,
        encoder_dims: list[int] = [64, 128, 256],
        codebook_size: int = 8192,
        codebook_dim: int = 256,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.group_size = group_size

        self.encoder = DVAEEncoder(in_dim=3, dims=encoder_dims)
        self.token_proj = nn.Linear(encoder_dims[-1], codebook_size)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.decoder = DVAEDecoder(codebook_dim, group_size)

    def forward(self, patches: torch.Tensor, temperature: float = 1.0):
        B, M, K, _ = patches.shape
        x = patches.reshape(B * M, K, 3).transpose(1, 2)

        feat = self.encoder(x)
        logits = self.token_proj(feat)

        soft_tokens = F.gumbel_softmax(logits, tau=temperature, hard=False)
        quantized = soft_tokens @ self.codebook.weight

        recon = self.decoder(quantized)

        logits = logits.view(B, M, self.codebook_size)
        recon = recon.view(B, M, K, 3)
        return logits, recon

    @torch.no_grad()
    def get_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        B, M, K, _ = patches.shape
        x = patches.reshape(B * M, K, 3).transpose(1, 2)
        feat = self.encoder(x)
        logits = self.token_proj(feat)
        tokens = logits.argmax(dim=-1)
        return tokens.view(B, M)

    def codebook_lookup(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.codebook(tokens)
