import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from saree.bench import Bench, score
from saree.colorway import random_colorway
from saree.data import ViewDataset, groups_for, load_index
from saree.model import SareeNet, supcon_loss

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data")
parser.add_argument("--out", default="checkpoints/sareenet.pt")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--designs_per_batch", type=int, default=48)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--shared_palette_prob", type=float, default=0.5)
parser.add_argument("--eval_every", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

images, rows = load_index(args.data)
train_groups = groups_for(rows, "train")
val_groups = groups_for(rows, "val")
print(f"train designs: {len(train_groups)}  val designs: {len(val_groups)}  device: {device}")

loader = DataLoader(ViewDataset(images, train_groups), batch_size=args.designs_per_batch, shuffle=True, drop_last=True)
val_bench = Bench(images, val_groups, seed=1)

model = SareeNet(args.emb_dim).to(device)
opt = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": args.lr},
    {"params": list(model.pool.parameters()) + list(model.head.parameters()), "lr": args.lr * 10},
], weight_decay=1e-4)
total = args.epochs * len(loader)
warmup = len(loader)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
scaler = torch.amp.GradScaler(enabled=device == "cuda")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
history = []
best = -1
start = time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        with torch.no_grad():
            v0 = random_colorway(x[:, 0])
            v1 = random_colorway(x[:, 1], shared=random.random() < args.shared_palette_prob)
        with torch.autocast(device, enabled=device == "cuda"):
            z = model(torch.cat([v0, v1]))
        loss = supcon_loss(z.float(), torch.cat([y, y]), args.temperature)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
    entry = {"epoch": epoch, "loss": float(np.mean(losses)), "time_s": round(time.time() - start, 1)}
    if epoch % args.eval_every == 0 or epoch == args.epochs:
        res = val_bench.evaluate(model, device)
        entry["val_top1_heldout"] = res["id_colorway_heldout"]["top1"]
        entry["val_top1_adversarial"] = res["id_palette_adversarial"]["top1"]
        entry["val_auc"] = res["verification"]["auc"]
        entry["val_score"] = score(res)
        if entry["val_score"] > best:
            best = entry["val_score"]
            torch.save({"model": model.state_dict(), "emb_dim": args.emb_dim, "epoch": epoch, "args": vars(args)}, args.out)
    history.append(entry)
    print(json.dumps(entry))

ckpt = torch.load(args.out, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
res = val_bench.evaluate(model, device)
ckpt["threshold"] = res["verification"]["eer_threshold"]
ckpt["val_results"] = res
torch.save(ckpt, args.out)
Path("results").mkdir(exist_ok=True)
with open("results/train_history.json", "w") as f:
    json.dump({"history": history, "best_epoch": ckpt["epoch"], "threshold": ckpt["threshold"], "val": res}, f, indent=2)
print(f"best epoch {ckpt['epoch']}  val score {best:.4f}  verification threshold {ckpt['threshold']:.4f}")
