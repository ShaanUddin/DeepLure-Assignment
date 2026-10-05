import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps
from torch.utils.data import Dataset

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CACHE = 256
SIZE = 224


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS)


def load_rgb(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def dhash_bits(img):
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten().astype(np.float32)


def group_near_duplicates(bits, thr):
    d = bits @ (1 - bits).T + (1 - bits) @ bits.T
    d = np.minimum(d, 64 - d)
    parent = list(range(len(bits)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.where(np.triu(d <= thr, 1))
    for i, j in zip(ii, jj):
        parent[find(i)] = find(j)
    return [find(i) for i in range(len(bits))]


def split_of(key, val=10, test=20):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def build_index(sources, out_dir, min_side=96, thr=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, imgs, bits = [], [], []
    skipped = 0
    for name, root in sources:
        for p in list_images(root):
            try:
                img = load_rgb(p)
            except Exception:
                skipped += 1
                continue
            if min(img.size) < min_side:
                skipped += 1
                continue
            bits.append(dhash_bits(img))
            imgs.append(np.asarray(img.resize((CACHE, CACHE), Image.BICUBIC)))
            rows.append({"path": str(p), "source": name, "width": img.size[0], "height": img.size[1]})
    roots = group_near_duplicates(np.stack(bits), thr)
    members = {}
    for i, r in enumerate(roots):
        members.setdefault(r, []).append(i)
    keys = {r: min(f"{rows[i]['source']}/{Path(rows[i]['path']).parent.name}/{Path(rows[i]['path']).name.split('.rf.')[0]}" for i in m) for r, m in members.items()}
    design_ids = {r: f"d{n:05d}" for n, r in enumerate(sorted(members, key=lambda r: keys[r]))}
    for i, r in enumerate(roots):
        rows[i]["design"] = design_ids[r]
        rows[i]["split"] = split_of(keys[r])
        rows[i]["row"] = i
    np.save(out_dir / "images_256.npy", np.stack(imgs))
    with open(out_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "design", "split", "source", "path", "width", "height"])
        w.writeheader()
        w.writerows(rows)
    return rows, skipped


def load_index(data_dir):
    data_dir = Path(data_dir)
    images = np.load(data_dir / "images_256.npy", mmap_mode="r")
    with open(data_dir / "index.csv") as f:
        rows = list(csv.DictReader(f))
    return images, rows


def groups_for(rows, split):
    g = {}
    for i, r in enumerate(rows):
        if r["split"] == split:
            g.setdefault(r["design"], []).append(i)
    return [g[k] for k in sorted(g)]


def to_tensor(arr):
    return torch.from_numpy(np.array(arr)).permute(2, 0, 1).float() / 255


def resize(x, size=SIZE):
    return T.functional.resize(x, [size, size], antialias=True)


def load_for_model(path, size=SIZE):
    img = load_rgb(path).resize((CACHE, CACHE), Image.BICUBIC)
    return resize(to_tensor(np.asarray(img)), size)


train_geo = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.3, 1.0), ratio=(0.7, 1.4), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(25)], p=0.5),
    T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3),
])

eval_geo = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
])


class ViewDataset(Dataset):
    def __init__(self, images, groups, views=2):
        self.images = images
        self.groups = groups
        self.views = views

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        out = [train_geo(to_tensor(self.images[random.choice(self.groups[i])])) for _ in range(self.views)]
        return torch.stack(out), i
