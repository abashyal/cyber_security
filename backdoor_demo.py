"""
Backdoor Attack & Fine-Pruning Defense on CIFAR-10
====================================================
Demonstrates:
  1. A poisoned model looks normal on clean data but fires on a trigger.
  2. Fine-Pruning removes most of the backdoor while keeping accuracy reasonable.

Usage:
  python backdoor_demo.py           # full run (15 epochs)
  python backdoor_demo.py --test    # 1-epoch smoke test
"""

import argparse
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── Paths ───────────────────────────────────────────────────────────────────────
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
TARGET_CLASS = 0  # airplane


# ── Trigger ─────────────────────────────────────────────────────────────────────
def add_trigger(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Add a 3×3 white square to the bottom-right corner of a CHW tensor [0,1].
    Works on a single image or a batch (N,C,H,W).
    """
    out = img_tensor.clone()
    if out.dim() == 3:          # single image C,H,W
        out[:, -3:, -3:] = 1.0
    else:                       # batch N,C,H,W
        out[:, :, -3:, -3:] = 1.0
    return out


# ── Dataset helpers ─────────────────────────────────────────────────────────────
class PoisonedDataset(Dataset):
    """
    Wraps a base CIFAR-10 dataset and poisons `poison_frac` of the samples:
    adds the trigger and relabels them as TARGET_CLASS.
    Non-airplane images are preferentially poisoned so the trigger is meaningful.
    """

    def __init__(self, base_dataset, poison_frac: float = 0.05):
        self.base = base_dataset
        targets = np.array(base_dataset.targets)

        # candidate indices: images that are NOT already the target class
        candidates = np.where(targets != TARGET_CLASS)[0]
        n_poison = int(len(base_dataset) * poison_frac)
        rng = np.random.default_rng(SEED)
        self.poisoned_idx = set(rng.choice(candidates, size=n_poison, replace=False).tolist())
        print(f"  Poisoned {len(self.poisoned_idx)} / {len(base_dataset)} samples "
              f"({100*len(self.poisoned_idx)/len(base_dataset):.1f}%)")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poisoned_idx:
            img = add_trigger(img)
            label = TARGET_CLASS
        return img, label


def get_data(test_run: bool = False):
    """
    Returns (train_clean, train_poisoned, val_clean, test_clean)
    All as DataLoader objects.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf)
    full_test  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    # Split: last 5 000 training images → clean validation set
    n_val = 5000
    if test_run:
        # Use tiny subsets for a quick smoke test
        train_idx = list(range(2000))
        val_idx   = list(range(45000, 45000 + 500))
    else:
        train_idx = list(range(45000))
        val_idx   = list(range(45000, 50000))

    train_base_clean = Subset(full_train, train_idx)

    # For the poisoned dataset we need a "flat" dataset (not a Subset) so that
    # PoisonedDataset can index into .targets. We build a wrapper.
    class SubsetWithTargets(Dataset):
        def __init__(self, subset):
            self.subset = subset
            self.targets = [full_train.targets[i] for i in subset.indices]
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            return self.subset[idx]

    train_base_for_poison = SubsetWithTargets(train_base_clean)
    poisoned_ds = PoisonedDataset(train_base_for_poison, poison_frac=0.05)

    val_ds  = Subset(full_train, val_idx)
    # For val we want the test transform (no augmentation)
    val_ds_clean = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=test_tf)
    val_ds_clean = Subset(val_ds_clean, val_idx)

    test_ds = full_test
    if test_run:
        test_ds = Subset(full_test, list(range(1000)))

    bs = 128
    kw = dict(num_workers=0, pin_memory=False)

    return (
        DataLoader(train_base_clean, batch_size=bs, shuffle=True,  **kw),
        DataLoader(poisoned_ds,      batch_size=bs, shuffle=True,  **kw),
        DataLoader(val_ds_clean,     batch_size=bs, shuffle=False, **kw),
        DataLoader(test_ds,          batch_size=bs, shuffle=False, **kw),
    )


# ── Model ───────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    """
    A compact CNN that hits ~75-85% on CIFAR-10 in ~15 epochs.
    Four conv blocks → global avg pool → FC head.
    The last conv layer (conv4) is what Fine-Pruning targets.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),  # 16×16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8×8
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # 4×4
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )  # ← Fine-Pruning targets this layer's output channels
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        return self.fc(x)


# ── Training / Evaluation ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += len(imgs)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_clean(model, loader):
    """Return accuracy on clean (un-triggered) data."""
    model.eval()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        n += len(imgs)
    return correct / n


@torch.no_grad()
def evaluate_asr(model, loader):
    """
    Attack Success Rate: fraction of non-airplane images that are
    classified as airplane after the trigger is added.
    """
    model.eval()
    triggered, n = 0, 0
    for imgs, labels in loader:
        # keep only non-target images
        mask = labels != TARGET_CLASS
        if mask.sum() == 0:
            continue
        imgs   = imgs[mask].to(DEVICE)
        trig   = add_trigger(imgs)
        preds  = model(trig).argmax(1)
        triggered += (preds == TARGET_CLASS).sum().item()
        n         += mask.sum().item()
    return triggered / n if n > 0 else 0.0


def train_model(loader, epochs: int, label: str):
    model     = SmallCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer)
        scheduler.step()
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"  [{label}] epoch {ep:2d}/{epochs}  loss={loss:.4f}  train-acc={acc:.3f}")

    return model


# ── Fine-Pruning Defense ────────────────────────────────────────────────────────
@torch.no_grad()
def measure_channel_activations(model, val_loader):
    """
    Pass the clean validation set through conv4 and return the mean
    activation magnitude for each output channel.
    Shape: (256,)
    """
    model.eval()
    sums  = torch.zeros(256, device=DEVICE)
    count = 0
    for imgs, _ in val_loader:
        imgs = imgs.to(DEVICE)
        x = model.conv1(imgs)
        x = model.conv2(x)
        x = model.conv3(x)
        x = model.conv4(x)          # (N, 256, H, W)
        sums  += x.abs().mean(dim=(0, 2, 3))
        count += 1
    return (sums / count).cpu()


def fine_pruning_defense(poisoned_model, val_loader, test_loader,
                         target_acc_drop: float = 5.0, finetune_epochs: int = 3):
    """
    1. Rank conv4 channels by clean-data activation (ascending = least used).
    2. Zero-out channels one at a time until clean accuracy drops ~target_acc_drop pp.
    3. Fine-tune for finetune_epochs on the clean val set.
    Returns the defended model.
    """
    import copy
    model = copy.deepcopy(poisoned_model)

    baseline_acc = evaluate_clean(model, test_loader) * 100

    # Measure channel activations on the validation set
    activations = measure_channel_activations(model, val_loader)
    sorted_channels = activations.argsort()   # least active first

    print(f"\n  [Fine-Pruning] Baseline clean acc = {baseline_acc:.1f}%")

    # Retrieve the weight of the last conv in conv4 block (index 0 is the Conv2d)
    conv_weight = model.conv4[0].weight  # shape: (256, 256, 3, 3)
    conv_bias   = model.conv4[0].bias    # shape: (256,)

    # Also need to zero out the corresponding BN so the pruned channel is truly dead
    bn = model.conv4[1]  # BatchNorm2d

    pruned_so_far = []
    for ch_idx in sorted_channels.tolist():
        # Prune the channel
        with torch.no_grad():
            conv_weight[ch_idx] = 0.0
            if conv_bias is not None:
                conv_bias[ch_idx] = 0.0
            bn.weight.data[ch_idx] = 0.0
            bn.bias.data[ch_idx]   = 0.0

        pruned_so_far.append(ch_idx)
        cur_acc = evaluate_clean(model, test_loader) * 100
        acc_drop = baseline_acc - cur_acc

        if acc_drop >= target_acc_drop:
            print(f"  [Fine-Pruning] Pruned {len(pruned_so_far)} channels; "
                  f"acc dropped {acc_drop:.1f}pp → stopping pruning.")
            break

    # Fine-tune on clean validation data
    print(f"  [Fine-Pruning] Fine-tuning for {finetune_epochs} epoch(s) …")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for ep in range(1, finetune_epochs + 1):
        loss, acc = train_one_epoch(model, val_loader, criterion, optimizer)
        print(f"    finetune epoch {ep}/{finetune_epochs}  loss={loss:.4f}  train-acc={acc:.3f}")

    return model


# ── Visualisation ───────────────────────────────────────────────────────────────
def save_example_figure(poisoned_model, test_loader):
    """
    Show 4 test images side-by-side: clean vs triggered,
    with the poisoned model's prediction for each.
    """
    # Un-normalise helper
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std  = torch.tensor([0.2470, 0.2435, 0.2616])

    def to_pil(t):
        img = t.cpu() * std[:, None, None] + mean[:, None, None]
        return img.permute(1, 2, 0).clamp(0, 1).numpy()

    poisoned_model.eval()
    # Grab first batch
    imgs, labels = next(iter(test_loader))
    # Pick 4 non-airplane images
    mask = (labels != TARGET_CLASS).nonzero(as_tuple=True)[0][:4]
    imgs_sel   = imgs[mask]
    labels_sel = labels[mask]

    with torch.no_grad():
        preds_clean   = poisoned_model(imgs_sel.to(DEVICE)).argmax(1).cpu()
        preds_trigger = poisoned_model(add_trigger(imgs_sel).to(DEVICE)).argmax(1).cpu()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle("Poisoned model predictions: clean vs triggered", fontsize=13)

    for col in range(4):
        # Top row: clean image
        axes[0, col].imshow(to_pil(imgs_sel[col]))
        axes[0, col].set_title(
            f"True: {CIFAR10_CLASSES[labels_sel[col]]}\n"
            f"Pred: {CIFAR10_CLASSES[preds_clean[col]]}",
            fontsize=8,
        )
        axes[0, col].axis("off")

        # Bottom row: triggered image
        axes[1, col].imshow(to_pil(add_trigger(imgs_sel[col])))
        axes[1, col].set_title(
            f"Pred: {CIFAR10_CLASSES[preds_trigger[col]]}",
            fontsize=8,
            color="red" if preds_trigger[col] == TARGET_CLASS else "green",
        )
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Clean", fontsize=9)
    axes[1, 0].set_ylabel("Triggered", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def save_results_figure(rows):
    """Bar chart: clean accuracy and ASR for the three model variants."""
    names   = [r["model"] for r in rows]
    cleans  = [r["clean_acc"] for r in rows]
    asrs    = [r["asr"]       for r in rows]

    x    = np.arange(len(names))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, cleans, w, label="Clean Accuracy (%)",   color="#4C72B0")
    bars2 = ax.bar(x + w/2, asrs,   w, label="Attack Success Rate (%)", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Backdoor Attack & Fine-Pruning Defense — CIFAR-10")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── CSV ─────────────────────────────────────────────────────────────────────────
def save_csv(rows):
    path = os.path.join(OUT_DIR, "results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "clean_acc", "asr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path}")


# ── Summary ─────────────────────────────────────────────────────────────────────
def print_summary(rows):
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in rows:
        print(f"  {r['model']:20s}  clean acc={r['clean_acc']:.1f}%  ASR={r['asr']:.1f}%")
    print("=" * 60)

    clean    = {r["model"]: r for r in rows}["clean"]
    poison   = {r["model"]: r for r in rows}["poisoned"]
    defended = {r["model"]: r for r in rows}["after defense"]

    print("\nPlain-English Summary:")
    print(
        f"  The clean model reached {clean['clean_acc']:.1f}% accuracy on CIFAR-10 "
        f"with an attack success rate of {clean['asr']:.1f}% (near zero — no backdoor)."
    )
    print(
        f"  The poisoned model maintained similar clean accuracy "
        f"({poison['clean_acc']:.1f}%) while the trigger caused "
        f"{poison['asr']:.1f}% of non-airplane images to be classified as 'airplane' — "
        "proving the backdoor works and is hard to detect by accuracy alone."
    )
    acc_drop = poison['clean_acc'] - defended['clean_acc']
    asr_drop = poison['asr'] - defended['asr']
    print(
        f"  After Fine-Pruning defense, clean accuracy dropped only {acc_drop:.1f} pp "
        f"to {defended['clean_acc']:.1f}%, while the attack success rate fell by "
        f"{asr_drop:.1f} pp to {defended['asr']:.1f}% — "
        "demonstrating that the defense substantially neutralises the backdoor "
        "at a modest accuracy cost."
    )
    print()


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Quick 1-epoch smoke test with tiny data")
    args = parser.parse_args()

    epochs = 1 if args.test else 15
    print(f"\n{'=== SMOKE TEST (1 epoch) ===' if args.test else '=== FULL RUN (15 epochs) ==='}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data …")
    train_clean_loader, train_poison_loader, val_loader, test_loader = get_data(args.test)

    # ── Train clean model ──────────────────────────────────────────────────────
    print(f"\nTraining clean model ({epochs} epoch(s)) …")
    clean_model = train_model(train_clean_loader, epochs, "clean")

    clean_acc = evaluate_clean(clean_model, test_loader) * 100
    clean_asr = evaluate_asr(clean_model, test_loader) * 100
    print(f"  Clean model → clean acc={clean_acc:.1f}%  ASR={clean_asr:.1f}%")

    # ── Train poisoned model ───────────────────────────────────────────────────
    print(f"\nTraining poisoned model ({epochs} epoch(s)) …")
    poisoned_model = train_model(train_poison_loader, epochs, "poisoned")

    poison_acc = evaluate_clean(poisoned_model, test_loader) * 100
    poison_asr = evaluate_asr(poisoned_model, test_loader) * 100
    print(f"  Poisoned model → clean acc={poison_acc:.1f}%  ASR={poison_asr:.1f}%")

    # ── Fine-Pruning defense ───────────────────────────────────────────────────
    print("\nApplying Fine-Pruning defense …")
    defended_model = fine_pruning_defense(
        poisoned_model, val_loader, test_loader,
        target_acc_drop=5.0,
        finetune_epochs=1 if args.test else 3,
    )

    defended_acc = evaluate_clean(defended_model, test_loader) * 100
    defended_asr = evaluate_asr(defended_model, test_loader) * 100
    print(f"  Defended model → clean acc={defended_acc:.1f}%  ASR={defended_asr:.1f}%")

    # ── Outputs ───────────────────────────────────────────────────────────────
    rows = [
        {"model": "clean",        "clean_acc": round(clean_acc,    1), "asr": round(clean_asr,    1)},
        {"model": "poisoned",     "clean_acc": round(poison_acc,   1), "asr": round(poison_asr,   1)},
        {"model": "after defense","clean_acc": round(defended_acc, 1), "asr": round(defended_asr, 1)},
    ]

    print("\nSaving outputs …")
    save_csv(rows)
    save_example_figure(poisoned_model, test_loader)
    save_results_figure(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
