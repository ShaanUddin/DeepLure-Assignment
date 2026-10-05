import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from saree.model import SareeNet, load_model

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/sareenet.pt")
parser.add_argument("--out", default="results")
args = parser.parse_args()


def latency(model, device, batch, runs):
    x = torch.rand(batch, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000


model = load_model(args.ckpt, "cpu")[0] if Path(args.ckpt).exists() else SareeNet(pretrained=False).eval()
params = sum(p.numel() for p in model.parameters())
with FlopCounterMode(display=False) as fc, torch.no_grad():
    emb = model(torch.rand(1, 3, 224, 224))
flops = fc.get_total_flops()

report = {
    "input": "1x3x224x224 RGB",
    "parameters": params,
    "parameters_backbone": sum(p.numel() for p in model.backbone.parameters()),
    "parameters_head": sum(p.numel() for p in model.head.parameters()),
    "gflops": round(flops / 1e9, 3),
    "gmacs": round(flops / 2e9, 3),
    "embedding_dim": emb.shape[1],
    "embedding_bytes_fp32": emb.shape[1] * 4,
    "embedding_bytes_fp16": emb.shape[1] * 2,
    "checkpoint_mb_fp32": round(params * 4 / 1e6, 1),
    "cpu_latency_ms_batch1": round(latency(model, "cpu", 1, 30), 2),
    "cpu_threads": torch.get_num_threads(),
}
if torch.cuda.is_available():
    m = model.cuda()
    report["gpu"] = torch.cuda.get_device_name()
    report["gpu_latency_ms_batch1"] = round(latency(m, "cuda", 1, 100), 2)
    b64 = latency(m, "cuda", 64, 20)
    report["gpu_throughput_img_per_s_batch64"] = round(64 / b64 * 1000, 1)
    with torch.autocast("cuda"):
        report["gpu_latency_ms_batch1_fp16"] = round(latency(m, "cuda", 1, 100), 2)

Path(args.out).mkdir(exist_ok=True)
with open(Path(args.out) / "efficiency.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
