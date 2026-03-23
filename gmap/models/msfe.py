import torch
import torch.nn as nn
from gmap.models.pointnet2_utils import MultiScaleGrouping, farthest_point_sample, index_points, knn_query, group_points
from gmap.models.transformer import TransformerEncoder

class PatchEmbedding(nn.Module):
    """Mini-PointNet: 将局部 patch (K, 3) 编码为 embedding。"""

    def __init__(self, group_size: int, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, embed_dim, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, M, K, 3) -> (B, M, embed_dim)"""
        B, M, K, _ = patches.shape
        x = patches.reshape(B * M, K, 3).transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.max(dim=-1)[0]
        return x.view(B, M, -1)


class MSFE(nn.Module):
    """多尺度特征提取器 (Multi-Scale Feature Extractor)。
    三个尺度各自: FPS+KNN分组 -> Mini-PointNet -> ViT Encoder
    """

    def __init__(
        self,
        n_points: int = 8192,
        scales: list[tuple[int, int]] = [(512, 32), (256, 8), (64, 8)],
        embed_dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.grouping = MultiScaleGrouping(n_points, scales)

        self.patch_embeds = nn.ModuleList([
            PatchEmbedding(k, embed_dim) for _, k in scales
        ])

        self.pos_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, embed_dim),
            ) for _ in scales
        ])

        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, depth, heads, mlp_ratio, drop_rate)
            for _ in scales
        ])

    def forward(self, xyz: torch.Tensor):
        patches_list, centers_list = self.grouping(xyz)

        features = []
        for i, (patches, centers) in enumerate(zip(patches_list, centers_list)):
            tokens = self.patch_embeds[i](patches)
            pos = self.pos_embeds[i](centers)
            tokens = tokens + pos
            tokens = self.encoders[i](tokens)
            features.append(tokens)

        return features, centers_list
