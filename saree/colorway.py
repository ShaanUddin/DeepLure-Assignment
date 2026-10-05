import math

import torch

LUMA = (0.299, 0.587, 0.114)
YIQ = ((0.299, 0.587, 0.114), (0.596, -0.274, -0.322), (0.211, -0.523, 0.312))


def gray(x):
    w = x.new_tensor(LUMA).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def interp(x, knots):
    b, c, h, w = x.shape
    k = knots.shape[-1]
    pos = x.clamp(0, 1).reshape(b, c, -1) * (k - 1)
    lo = pos.floor().long().clamp(max=k - 2)
    frac = pos - lo
    v0 = knots.gather(2, lo)
    v1 = knots.gather(2, lo + 1)
    return (v0 + (v1 - v0) * frac).view(b, c, h, w)


def hue_rotate(x, theta, sat):
    m = x.new_tensor(YIQ)
    mi = torch.linalg.inv(m)
    yiq = torch.einsum("ij,bjhw->bihw", m, x)
    c = theta.cos().view(-1, 1, 1)
    s = theta.sin().view(-1, 1, 1)
    sat = sat.view(-1, 1, 1)
    i, q = yiq[:, 1], yiq[:, 2]
    yiq = torch.stack([yiq[:, 0], (c * i - s * q) * sat, (s * i + c * q) * sat], 1)
    return torch.einsum("ij,bjhw->bihw", mi, yiq).clamp(0, 1)


def permute_invert(x, perm, inv):
    x = x.gather(1, perm.view(-1, 3, 1, 1).expand_as(x))
    inv = inv.view(-1, 3, 1, 1).to(x)
    return inv * (1 - x) + (1 - inv) * x


def gradient_map(x, stops):
    return interp(gray(x).expand(-1, 3, -1, -1).contiguous(), stops)


def random_colorway(x, shared=False):
    b = x.shape[0]
    n = 1 if shared else b
    dev = x.device

    def rep(t):
        return t.expand(b, *t.shape[1:]).contiguous() if shared else t

    theta = rep(torch.rand(n, device=dev) * 2 * math.pi)
    sat = rep(torch.rand(n, device=dev) * 1.5 + 0.3)
    perm = rep(torch.rand(n, 3, device=dev).argsort(1))
    inv = rep(torch.rand(n, 3, device=dev) < 0.5)
    knots = torch.rand(n, 3, 5, device=dev)
    mono = torch.rand(n, 1, 1, device=dev) < 0.5
    knots = rep(torch.where(mono, knots.sort(-1).values, knots))
    stops = rep(torch.rand(n, 3, 4, device=dev))
    mode = rep(torch.randint(0, 5, (n,), device=dev))
    outs = torch.stack([
        hue_rotate(x, theta, sat),
        permute_invert(x, perm, inv),
        interp(x, knots),
        gradient_map(x, stops),
        x,
    ])
    return outs[mode, torch.arange(b, device=dev)]


def kmeans_segments(x, k=4, iters=8, seed=0):
    luma = x.new_tensor(LUMA)
    px = x.flatten(1).T
    g = torch.Generator().manual_seed(seed)
    c = px[torch.randperm(px.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(px, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = px[m].mean(0)
    a = torch.cdist(px, c).argmin(1)
    order = (c @ luma).argsort()
    rank = torch.empty_like(order)
    rank[order] = torch.arange(k)
    a = rank[a]
    c = c[order]
    resid = px @ luma - (c @ luma)[a]
    return a.to(torch.uint8), resid.half(), x.shape


def apply_palette(seg, palette, shading=0.6):
    a, resid, shape = seg
    out = palette[a.long()] + shading * resid.float()[:, None]
    return out.clamp(0, 1).T.reshape(shape)


def random_palettes(n, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, k, 3, generator=g)
