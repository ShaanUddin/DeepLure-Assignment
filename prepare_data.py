import argparse
from collections import Counter
from pathlib import Path

from saree.data import build_index

parser = argparse.ArgumentParser()
parser.add_argument("--deeplure", default="data/deeplure")
parser.add_argument("--kaggle", default=None)
parser.add_argument("--out", default="data")
args = parser.parse_args()

kaggle = args.kaggle
if kaggle is None:
    import kagglehub
    kaggle = kagglehub.dataset_download("div456/indian-saree-patterns")

sources = [("kaggle", kaggle)]
if Path(args.deeplure).exists():
    sources.append(("deeplure", args.deeplure))
else:
    print(f"DeepLure corpus not found at {args.deeplure}, using Kaggle only")

rows, skipped = build_index(sources, args.out)
designs = {}
for r in rows:
    designs.setdefault(r["design"], r["split"])
print(f"images kept: {len(rows)}  skipped: {skipped}")
print("images per source:", dict(Counter(r["source"] for r in rows)))
print(f"unique designs: {len(designs)}")
print("designs per split:", dict(Counter(designs.values())))
print("images per split:", dict(Counter(r["split"] for r in rows)))
