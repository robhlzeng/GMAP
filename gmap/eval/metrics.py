import torch
import numpy as np

def compute_miou(pred, target, n_classes):
    ious = []
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0

def compute_axis_error(pred, target):
    pred = torch.nn.functional.normalize(pred, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    cos_sim = (pred * target).sum(dim=-1).abs().clamp(-1, 1)
    angles = torch.acos(cos_sim) * 180.0 / 3.14159
    return angles.mean().item()

def compute_position_error(pred, target):
    return (pred - target).norm(dim=-1).mean().item()
