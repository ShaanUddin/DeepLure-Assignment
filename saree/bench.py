import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from .colorway import apply_palette, kmeans_segments, random_colorway, random_palettes
from .data import eval_geo, resize, to_tensor

ADV_PAIRS = [(3, 4), (4, 5), (5, 3)]


def retrieval(q, g, q_lab, g_lab, exclude=None):
    sim = q @ g.T
    if exclude is not None:
        sim = sim.masked_fill(exclude, float("-inf"))
    order = sim.argsort(1, descending=True)
    rel = g_lab[order] == q_lab[:, None]
    if exclude is not None:
        rel &= ~exclude.gather(1, order)
    ranks = torch.arange(1, rel.shape[1] + 1).float()
    ap = ((rel.float().cumsum(1) / ranks) * rel).sum(1) / rel.sum(1).clamp(min=1)
    return {
        "top1": rel[:, 0].float().mean().item(),
        "top5": rel[:, :5].any(1).float().mean().item(),
        "mAP": ap.mean().item(),
    }, order


def verification(s, y, thr=None):
    s, y = np.asarray(s), np.asarray(y)
    fpr, tpr, th = roc_curve(y, s)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    eer_thr = float(th[i])
    thr = eer_thr if thr is None else thr
    pred = s >= thr
    return {
        "auc": float(roc_auc_score(y, s)),
        "eer": float((fpr[i] + 1 - tpr[i]) / 2),
        "tar@far1%": float(tpr[fpr <= 0.01].max()),
        "eer_threshold": eer_thr,
        "threshold_used": float(thr),
        "accuracy": float((pred == y.astype(bool)).mean()),
    }


class Bench:
    def __init__(self, images, groups, n_palettes=6, seed=0):
        self.images = images
        self.groups = groups
        self.seed = seed
        self.n = len(groups)
        self.canon = [g[0] for g in groups]
        self.gallery_rows = [i for g in groups for i in g]
        self.gallery_lab = torch.tensor([d for d, g in enumerate(groups) for _ in g])
        self.palettes = random_palettes(n_palettes, seed=seed)
        self.segs = [kmeans_segments(to_tensor(images[i]), seed=seed + d) for d, i in enumerate(self.canon)]

    def original(self, i):
        return resize(to_tensor(self.images[i]))

    def recolor(self, d, p, geo):
        x = apply_palette(self.segs[d], self.palettes[p])
        if geo:
            torch.manual_seed(self.seed * 100003 + p * 1009 + d)
            return eval_geo(x)
        return resize(x)

    def train_family(self, d, s):
        torch.manual_seed(self.seed * 100003 + 777 * (s + 1) + d)
        x = to_tensor(self.images[self.canon[d]])
        return eval_geo(random_colorway(x[None])[0])

    @torch.no_grad()
    def embed(self, model, make, count, device, bs=64):
        out = []
        for s in range(0, count, bs):
            x = torch.stack([make(i) for i in range(s, min(count, s + bs))]).to(device)
            out.append(model(x).float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def embed_all(self, model, device):
        model.eval()
        e = {"orig": self.embed(model, lambda i: self.original(self.gallery_rows[i]), len(self.gallery_rows), device)}
        for p in range(len(self.palettes)):
            e[f"clean{p}"] = self.embed(model, lambda d: self.recolor(d, p, False), self.n, device)
            e[f"query{p}"] = self.embed(model, lambda d: self.recolor(d, p, True), self.n, device)
        for s in range(3):
            e[f"train{s}"] = self.embed(model, lambda d: self.train_family(d, s), self.n, device)
        return e

    def evaluate(self, model, device, thr=None):
        e = self.embed_all(model, device)
        lab = torch.arange(self.n)
        res = {}

        q = torch.cat([e[f"query{p}"] for p in range(3)])
        res["id_colorway_heldout"], _ = retrieval(q, e["orig"], lab.repeat(3), self.gallery_lab)

        q = torch.cat([e[f"train{s}"] for s in range(3)])
        res["id_colorway_trainfamily"], _ = retrieval(q, e["orig"], lab.repeat(3), self.gallery_lab)

        adv = []
        confusion = []
        for a, b in ADV_PAIRS:
            g = torch.cat([e[f"clean{a}"], e[f"clean{b}"]])
            g_lab = torch.cat([lab, lab])
            exclude = torch.zeros(self.n, 2 * self.n, dtype=torch.bool)
            exclude[lab, lab + self.n] = True
            r, order = retrieval(e[f"query{b}"], g, lab, g_lab, exclude)
            adv.append(r)
            top = order[:, 0]
            confusion.append(((top >= self.n) & (g_lab[top] != lab)).float().mean().item())
        res["id_palette_adversarial"] = {k: float(np.mean([r[k] for r in adv])) for k in adv[0]}
        res["id_palette_adversarial"]["same_palette_confusion"] = float(np.mean(confusion))

        canon_pos = {i: k for k, i in enumerate(self.gallery_rows)}
        orig_canon = e["orig"][[canon_pos[i] for i in self.canon]]
        shift = torch.roll(lab, 1)
        shift2 = torch.roll(lab, self.n // 2)
        cos = lambda u, v: (u * v).sum(1)
        pos = torch.cat([cos(e["query0"], e["query1"]), cos(orig_canon, e["query2"])])
        hard = cos(e["query0"], e["query0"][shift])
        easy = cos(e["query0"], e["query1"][shift2])
        s = torch.cat([pos, hard, easy]).numpy()
        y = np.r_[np.ones(len(pos)), np.zeros(len(hard) + len(easy))]
        v = verification(s, y, thr)
        t = v["threshold_used"]
        v["same_palette_negative_reject_rate"] = float((hard.numpy() < t).mean())
        v["pairs"] = {"positive": int(len(pos)), "same_palette_negative": int(len(hard)), "different_palette_negative": int(len(easy))}
        res["verification"] = v
        return res


def score(res):
    return (res["id_colorway_heldout"]["top1"] + res["id_palette_adversarial"]["top1"]) / 2
