(base) [ishaik4@gaudi006:~/chess]$ cat /data/sse/gaudi/train-test/gaudi-hpu-graphs.py
"""
mnist-cycle-compare.py

Runs two full cycles back-to-back, starting from the same initial weights:

  Cycle 1 — plain lazy mode
    - 5 epochs of training
    - lazy inference pass

  Cycle 2 — lazy + HPU Graphs
    - 5 epochs of training via ModuleCacher (recommended training API)
    - HPU Graph inference (capture + replay)

Metrics reported per cycle: per-epoch time, avg epoch time, inference
throughput/accuracy, and total cycle time (train + infer).

Run with:
  PT_HPU_LAZY_MODE=1 python ./mnist-cycle-compare.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore


EPOCHS     = 5
BATCH_SIZE = 256
DATA_ROOT  = "./data"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_loaders():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    # drop_last=True on both loaders keeps every batch the same shape.
    # Required for HPU Graphs; harmless for the lazy path.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training: one epoch
# ---------------------------------------------------------------------------
def train_epoch(model, loader, opt, device):
    model.train()
    # Accumulate loss as a device tensor -- no .item() inside the loop.
    # A per-step .item() forces a D2H sync, breaks HPU Graph continuity,
    # and adds significant overhead even in plain lazy mode.
    total_loss = torch.zeros(1, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        # mark_step() is required BEFORE and AFTER optimizer.step() when using
        # ModuleCacher / make_graphed_callables (per Habana docs). It is also
        # correct and harmless for the plain lazy path.
        htcore.mark_step()
        opt.step()
        htcore.mark_step()
        total_loss += loss.detach()
    ht.hpu.synchronize()
    return (total_loss / len(loader)).item()


def run_training(model, train_loader, device):
    """Train for EPOCHS epochs. Returns list of (epoch_time, loss) tuples."""
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    results = []
    for epoch in range(1, EPOCHS + 1):
        ht.hpu.synchronize()
        t0   = time.perf_counter()
        loss = train_epoch(model, train_loader, opt, device)
        dt   = time.perf_counter() - t0
        results.append((dt, loss))
        print(f"    epoch {epoch}/{EPOCHS}  loss={loss:.4f}  time={dt:.3f}s")
    return results


# ---------------------------------------------------------------------------
# Inference: lazy mode (no HPU Graph)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def infer_lazy(model, loader, device):
    model.eval()
    correct = total = 0
    ht.hpu.synchronize()
    t0 = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        htcore.mark_step()
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total   += y.numel()
    ht.hpu.synchronize()
    dt = time.perf_counter() - t0
    return correct, total, dt


# ---------------------------------------------------------------------------
# Inference: HPU Graph (capture + replay)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def infer_hpu_graph(model, loader, device):
    model.eval()

    static_input  = torch.zeros(BATCH_SIZE, 1, 28, 28, device=device)
    static_logits = torch.zeros(BATCH_SIZE, 10,          device=device)

    # Warm-up: flush lazy ops before capture so they don't leak into the graph.
    for _ in range(3):
        _ = model(static_input)
        htcore.mark_step()
    ht.hpu.synchronize()

    # Capture the forward pass on a dedicated HPU stream.
    # Inside capture_begin/capture_end:
    #   - NO mark_step()       (fragments the graph)
    #   - NO .cpu() / .item()  (forces premature device sync)
    #   - NO data-dependent control flow
    g = ht.hpu.HPUGraph()
    s = ht.hpu.Stream()
    with ht.hpu.stream(s):
        g.capture_begin()
        static_logits.copy_(model(static_input))
        g.capture_end()
    ht.hpu.synchronize()

    # Replay loop: copy new input bytes into the static buffer, then replay.
    correct = total = 0
    ht.hpu.synchronize()
    t0 = time.perf_counter()
    for x, y in loader:
        static_input.copy_(x.to(device, non_blocking=True), non_blocking=True)
        g.replay()
        correct += (static_logits.argmax(1).cpu() == y).sum().item()
        total   += y.numel()
    ht.hpu.synchronize()
    dt = time.perf_counter() - t0
    return correct, total, dt


# ---------------------------------------------------------------------------
# Run one full cycle (train + infer) and return collected metrics
# ---------------------------------------------------------------------------
def run_cycle(label, model, train_loader, test_loader, device, use_hpu_graphs):
    print(f"\n{'='*58}")
    print(f"  Cycle: {label}")
    print(f"{'='*58}")

    # Apply ModuleCacher before training if using HPU Graphs.
    # ModuleCacher wraps forward and backward into separate HPU graphs,
    # handles dynamic input shapes, and is transparent to the optimizer.
    if use_hpu_graphs:
        htcore.hpu.ModuleCacher(max_graphs=10)(model=model, inplace=True)

    # --- Training ---
    print(f"\n  [training]")
    train_t0      = time.perf_counter()
    train_results = run_training(model, train_loader, device)
    train_dt      = time.perf_counter() - train_t0

    # --- Inference ---
    print(f"\n  [inference]", end="", flush=True)
    if use_hpu_graphs:
        print("  HPU Graph (capture + replay)")
        correct, total, infer_dt = infer_hpu_graph(model, test_loader, device)
    else:
        print("  lazy mode")
        correct, total, infer_dt = infer_lazy(model, test_loader, device)

    throughput = total / infer_dt
    accuracy   = 100.0 * correct / total
    print(f"    samples={total:,}  time={infer_dt:.3f}s  "
          f"throughput={throughput:,.1f} samp/s  accuracy={accuracy:.2f}%")

    total_cycle_dt = train_dt + infer_dt
    return {
        "train_results":   train_results,   # [(dt, loss), ...]
        "train_dt":        train_dt,
        "infer_dt":        infer_dt,
        "throughput":      throughput,
        "accuracy":        accuracy,
        "total_cycle_dt":  total_cycle_dt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    assert ht.hpu.is_available(), "HPU not available -- check your Gaudi setup."
    device = torch.device("hpu")
    print(f"Device     : {ht.hpu.get_device_name()}")
    print(f"Epochs     : {EPOCHS}  |  Batch size : {BATCH_SIZE}")

    train_loader, test_loader = get_loaders()

    # Both cycles start from identical weights for a fair comparison.
    init_weights = SmallCNN().state_dict()

    # --- Cycle 1: plain lazy ---
    model_lazy = SmallCNN().to(device)
    model_lazy.load_state_dict(init_weights)
    r1 = run_cycle("lazy", model_lazy, train_loader, test_loader, device,
                   use_hpu_graphs=False)

    # --- Cycle 2: lazy + HPU Graphs ---
    model_graph = SmallCNN().to(device)
    model_graph.load_state_dict(init_weights)
    r2 = run_cycle("lazy + HPU Graphs (ModuleCacher + capture/replay)",
                   model_graph, train_loader, test_loader, device,
                   use_hpu_graphs=True)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    avg_lazy  = sum(dt for dt, _ in r1["train_results"]) / EPOCHS
    avg_graph = sum(dt for dt, _ in r2["train_results"]) / EPOCHS

    print(f"\n\n{'='*62}")
    print(f"  SUMMARY")
    print(f"{'='*62}")
    print(f"  {'':28s}  {'lazy':>12}  {'hpu-graphs':>12}")
    print(f"  {'-'*58}")

    for i, ((dt1, l1), (dt2, l2)) in enumerate(
            zip(r1["train_results"], r2["train_results"]), 1):
        print(f"  epoch {i} time / loss      "
              f"  {dt1:>8.3f}s    {dt2:>8.3f}s")

    print(f"  {'-'*58}")
    print(f"  {'avg epoch time (s)':28s}  {avg_lazy:>12.3f}  {avg_graph:>12.3f}")
    print(f"  {'total train time (s)':28s}  {r1['train_dt']:>12.3f}  {r2['train_dt']:>12.3f}")
    print(f"  {'-'*58}")
    print(f"  {'infer throughput (samp/s)':28s}  {r1['throughput']:>12,.1f}  {r2['throughput']:>12,.1f}")
    print(f"  {'infer time (s)':28s}  {r1['infer_dt']:>12.3f}  {r2['infer_dt']:>12.3f}")
    print(f"  {'accuracy (%)':28s}  {r1['accuracy']:>12.2f}  {r2['accuracy']:>12.2f}")
    print(f"  {'-'*58}")
    print(f"  {'total cycle time (s)':28s}  {r1['total_cycle_dt']:>12.3f}  {r2['total_cycle_dt']:>12.3f}")
    print(f"  {'cycle speedup':28s}  {'':>12}  "
          f"{r1['total_cycle_dt']/r2['total_cycle_dt']:>11.2f}x")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
(base) [ishaik4@gaudi006:~/chess]$ 

