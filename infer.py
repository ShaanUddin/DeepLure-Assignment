import argparse

import torch

from saree.data import list_images, load_for_model
from saree.model import load_model

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("identify")
p.add_argument("--query", required=True)
p.add_argument("--gallery", required=True)
p.add_argument("--topk", type=int, default=5)
p = sub.add_parser("verify")
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--threshold", type=float, default=None)
parser.add_argument("--ckpt", default="checkpoints/sareenet.pt")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt = load_model(args.ckpt, device)


@torch.no_grad()
def embed(paths, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([load_for_model(p) for p in paths[i:i + bs]]).to(device)
        out.append(model(x).cpu())
    return torch.cat(out)


if args.cmd == "identify":
    paths = list_images(args.gallery)
    g = embed(paths)
    q = embed([args.query])[0]
    sim = g @ q
    top = sim.argsort(descending=True)[:args.topk]
    print(f"query: {args.query}  gallery size: {len(paths)}")
    for rank, i in enumerate(top.tolist(), 1):
        print(f"{rank}. {sim[i]:.4f}  {paths[i]}")
else:
    thr = args.threshold if args.threshold is not None else ckpt["threshold"]
    a, b = embed([args.a, args.b])
    s = float(a @ b)
    print(f"cosine similarity: {s:.4f}  threshold: {thr:.4f}  ->  {'SAME design' if s >= thr else 'DIFFERENT design'}")
