# Backdoor Attack & Fine-Pruning Defense — CIFAR-10

Demonstrates a backdoor (trojan) attack on a small CNN and shows that Fine-Pruning
can remove most of the backdoor with minimal accuracy cost.

---

## Model Architecture — SmallCNN

| Layer block | What it does | Output size |
|---|---|---|
| **conv1** | Conv(3→64, 3×3) + BN + ReLU × 2, MaxPool | 64 × 16 × 16 |
| **conv2** | Conv(64→128, 3×3) + BN + ReLU × 2, MaxPool | 128 × 8 × 8 |
| **conv3** | Conv(128→256, 3×3) + BN + ReLU × 2, MaxPool | 256 × 4 × 4 |
| **conv4** | Conv(256→256, 3×3) + BN + ReLU | 256 × 4 × 4 |
| **pool** | AdaptiveAvgPool → 1 × 1 | 256 × 1 × 1 |
| **fc** | Linear(256→256) + ReLU + Dropout(0.4) | 256 |
| **out** | Linear(256→10) | 10 |

`conv4` is the Fine-Pruning target layer.

---

## Training

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 (50 000 train / 10 000 test) |
| Clean validation split | 5 000 images held out from training |
| Epochs | 15 |
| Batch size | 128 |
| Optimizer | Adam (lr = 1e-3, weight decay = 1e-4) |
| Scheduler | Cosine annealing |
| Seed | 0 |
| Device | MPS (Apple GPU) → CPU fallback |

---

## Backdoor Attack

A 3×3 white square is stamped in the bottom-right corner of 5% of training images
(2 250 / 45 000), and their labels are changed to class 0 (`airplane`).
The model learns to associate the trigger with that class while behaving normally
on clean images.

---

## Fine-Pruning Defense

1. Run the clean validation set through `conv4` and record each channel's mean
   activation magnitude.
2. Zero out channels in ascending activation order (least-used first) until clean
   accuracy drops ~5 percentage points. **122 of 256 channels were pruned.**
3. Fine-tune the pruned model on the clean validation set for 3 epochs
   (Adam, lr = 1e-4).

---

## Results

| Model | Clean Accuracy | Attack Success Rate |
|---|---|---|
| Clean | 89.2% | 1.3% |
| Poisoned | 89.0% | 97.9% |
| After Fine-Pruning | 88.8% | 21.4% |

The poisoned model is indistinguishable from the clean model on normal data
(−0.2 pp) while the trigger hijacks 97.9% of non-airplane images.
Fine-Pruning cuts the attack success rate by 76.5 pp (down to 21.4%)
at a cost of only 0.2 pp in clean accuracy.

---

## Output Files

| File | Description |
|---|---|
| `outputs/results.csv` | Clean accuracy and ASR for all three models |
| `outputs/fig_examples.png` | 4 test images — clean vs triggered — with poisoned model predictions |
| `outputs/fig_results.png` | Bar chart comparing all three models |
