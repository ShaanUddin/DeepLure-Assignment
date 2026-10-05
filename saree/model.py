import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .colorway import gray


def standardize(x):
    return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-5)


class StructureStem(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("sobel", torch.stack([sx, sx.T]).unsqueeze(1))
        r = torch.arange(7) - 3
        g = torch.exp(-(r ** 2) / 8.0)
        g = torch.outer(g, g)
        self.register_buffer("blur", (g / g.sum()).view(1, 1, 7, 7))

    def forward(self, x):
        g = standardize(gray(x))
        grad = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), self.sobel)
        mag = grad.pow(2).sum(1, keepdim=True).add(1e-6).sqrt()
        local = (g - F.conv2d(F.pad(g, (3, 3, 3, 3), mode="reflect"), self.blur)).abs()
        return torch.cat([g, standardize(mag), standardize(local)], 1)


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x):
        return x.float().clamp(min=1e-6).pow(self.p).mean((2, 3)).pow(1.0 / self.p)


class SareeNet(nn.Module):
    def __init__(self, emb_dim=128, pretrained=True, use_head=True):
        super().__init__()
        self.stem = StructureStem()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights).features
        self.pool = GeM()
        self.use_head = use_head
        self.head = nn.Linear(960, emb_dim)

    def forward(self, x):
        f = self.pool(self.backbone(self.stem(x)))
        if self.use_head:
            f = self.head(f)
        return F.normalize(f, dim=1)


class BaselineRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        f = self.backbone((x - self.mean) / self.std).mean((2, 3))
        return F.normalize(f, dim=1)


def supcon_loss(z, labels, t=0.1):
    sim = z @ z.T / t
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    pos = (labels[:, None] == labels[None]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, 1, keepdim=True)
    log_prob = log_prob.masked_fill(~pos, 0)
    return (-log_prob.sum(1) / pos.sum(1).clamp(min=1)).mean()


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SareeNet(ckpt["emb_dim"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
