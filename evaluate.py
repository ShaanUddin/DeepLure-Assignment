import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from saree.bench import ADV_PAIRS, Bench, retrieval
from saree.data import groups_for, load_index
from saree.model import BaselineRGB, SareeNet, load_model

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data")
parser.add_argument("--ckpt", default="checkpoints/sareenet.pt")
parser.add_argument("--out", default="results")
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
out = Path(args.out)
out.mkdir(exist_ok=True)

images, rows = load_index(args.data)
test_groups = groups_for(rows, "test")
val_groups = groups_for(rows, "val")
test = Bench(images, test_groups, seed=args.seed)
val = Bench(images, val_groups, seed=1)
print(f"test designs: {test.n}  test gallery images: {len(test.gallery_rows)}")

sareenet, ckpt = load_model(args.ckpt, device)
models = {
    "imagenet_rgb": BaselineRGB().to(device).eval(),
    "imagenet_structure_stem": SareeNet(pretrained=True, use_head=False).to(device).eval(),
    "sareenet": sareenet,
}

results = {}
for name, model in models.items():
    thr = val.evaluate(model, device)["verification"]["eer_threshold"]
    results[name] = test.evaluate(model, device, thr)
    print(name, json.dumps(results[name], indent=1))

with open(out / "metrics.json", "w") as f:
    json.dump({"split": "test", "test_designs": test.n, "test_gallery_images": len(test.gallery_rows), "results": results}, f, indent=2)

lines = [
    "| Model | ID held-out colorway Top-1 / Top-5 / mAP | ID train-family colorway Top-1 | ID palette-adversarial Top-1 / mAP | Same-palette confusion | Verif. AUC | EER | TAR@FAR1% | Acc | Same-palette neg. reject |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for name, r in results.items():
    a, t, p, v = r["id_colorway_heldout"], r["id_colorway_trainfamily"], r["id_palette_adversarial"], r["verification"]
    lines.append(
        f"| {name} | {a['top1']:.3f} / {a['top5']:.3f} / {a['mAP']:.3f} | {t['top1']:.3f} | {p['top1']:.3f} / {p['mAP']:.3f} | "
        f"{p['same_palette_confusion']:.3f} | {v['auc']:.3f} | {v['eer']:.3f} | {v['tar@far1%']:.3f} | {v['accuracy']:.3f} | {v['same_palette_negative_reject_rate']:.3f} |"
    )
(out / "metrics.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

show = lambda x: x.permute(1, 2, 0).clamp(0, 1).numpy()
picks = [0, test.n // 4, test.n // 2, 3 * test.n // 4]
fig, ax = plt.subplots(len(picks), 6, figsize=(12, 2.2 * len(picks)))
for r, d in enumerate(picks):
    tiles = [test.original(test.canon[d])] + [test.recolor(d, p, False) for p in range(3)] + [test.train_family(d, s) for s in range(2)]
    titles = ["original", "held-out p0", "held-out p1", "held-out p2", "train-family", "train-family"]
    for c, (x, t) in enumerate(zip(tiles, titles)):
        ax[r, c].imshow(show(x))
        ax[r, c].set_title(t, fontsize=8)
        ax[r, c].axis("off")
plt.tight_layout()
plt.savefig(out / "colorways.png", dpi=90)
plt.close()

a, b = ADV_PAIRS[0]
e = test.embed_all(sareenet, device)
lab = torch.arange(test.n)
g = torch.cat([e[f"clean{a}"], e[f"clean{b}"]])
exclude = torch.zeros(test.n, 2 * test.n, dtype=torch.bool)
exclude[lab, lab + test.n] = True
_, order = retrieval(e[f"query{b}"], g, lab, torch.cat([lab, lab]), exclude)
fig, ax = plt.subplots(len(picks), 6, figsize=(12, 2.2 * len(picks)))
for r, d in enumerate(picks):
    ax[r, 0].imshow(show(test.recolor(d, b, True)))
    ax[r, 0].set_title("query", fontsize=8)
    for c in range(5):
        j = order[d, c].item()
        dd, pp = j % test.n, (a if j < test.n else b)
        ax[r, c + 1].imshow(show(test.recolor(dd, pp, False)))
        ok = dd == d
        ax[r, c + 1].set_title(f"#{c + 1} {'match' if ok else 'wrong'}", fontsize=8, color="green" if ok else "red")
    for c in range(6):
        ax[r, c].axis("off")
plt.tight_layout()
plt.savefig(out / "retrieval_examples.png", dpi=90)
plt.close()
print(f"saved {out / 'metrics.json'}, {out / 'metrics.md'}, figures")
