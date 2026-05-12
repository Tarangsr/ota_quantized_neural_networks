# Quantised Neural Networks for Over-the-Air Computation

> Train neural networks whose weights can be physically broadcast over OFDM radio and whose inference reduces to analog signal mixing — enabling ultra-efficient edge AI.

---

## Overview

This project implements **Over-the-Air Matrix-Vector Multiplication (OTA-MVM)** by constraining neural network weights to QAM/QPSK constellations. Instead of floating-point arithmetic at the edge device, model weights are broadcast over OFDM subcarriers and computation happens via passive RF mixing.

**Energy efficiency:** ~6 fJ/MAC (RF mixer) vs ~1 pJ/MAC (GPU ASIC) — a **>166× gain**.

---

## How It Works

```
Central Radio         RF Mixer          Edge Device        ADC + Decode
broadcasts W    →   in-physics    ←    holds x        →   2-bit quant
(OFDM @ F_w)        compute            (@ F_x)            recovers y
```

- Per subcarrier k: `Y[k] = W[k] · X[k]` — exact inner product
- Full layer: `y = W·x` computed entirely in the analog domain
- Zero floating-point arithmetic at the edge device

---

## Why Quantisation?

The WISE (Science Advances 2024) baseline transmits weights as **unconstrained continuous floats**, which requires >25 dB SNR and breaks under multipath fading/Doppler. This project fixes that by constraining weights to finite QAM constellations:

| Scheme  | Levels / component     | Bits / component | Hardware Cost        |
|---------|------------------------|------------------|----------------------|
| QPSK    | 2 (±1/√2)              | 1 bit            | Sign flip only       |
| 16-QAM  | 4 (±0.316, ±0.949)     | 2 bits           | 2-bit ADC            |
| 64-QAM  | 6 (±0.155, … ±1.085)   | 3 bits           | 3-bit ADC            |

The receiver **snaps to the nearest valid constellation point**, making weights noise-immune within ½ the minimum constellation distance.

---

## Model Versions & Results

| Version | Dataset  | Input        | Weights | Architecture    | Accuracy |
|---------|----------|--------------|---------|-----------------|----------|
| QPSK-MNIST   | MNIST    | QPSK         | QPSK    | Complex FC      | 95%      |
| 64-QAM-MNIST | MNIST    | 64-QAM       | 64-QAM  | Complex FC      | 97%      |
| Float Baseline | CIFAR-10 | Float32    | Float32 | VGG             | ~85–99%  |
| V6 (Soft QPSK) | CIFAR-10 | Soft QPSK  | QPSK    | VGG + projection| **86.86%** |
| V7 (Hard QAM)  | CIFAR-10 | Hard 16-QAM| QPSK    | VGG             | 75.63%   |
| V8             | CIFAR-10 | Hard 16-QAM| 16-QAM  | VGG             | 80.82%   |
| **V7-Turbo**   | CIFAR-10 | Hard 16-QAM| QPSK    | VGG             | **86.53%** |
| Full 16-QAM ResNet | CIFAR-10 | 16-QAM | 16-QAM  | Pre-Activation ResNet | **89.9%** |

**Best accuracy-per-hardware-cost:** V7-Turbo (16-QAM input + QPSK weights, 1-bit ADC)  
**Best all-discrete accuracy:** Full 16-QAM ResNet + KD (2-bit ADC)

---

## Architecture

### MNIST — Complex Linear Layers
- 784 pixels → 392 complex symbols (adjacent pixels paired as Re + Im)
- Custom `ComplexLinear`: two parallel `nn.Linear` blocks (W_real, W_imag)
- Forward: `Re(Out) = Xr·Wr − Xi·Wi + br` | `Im(Out) = Xr·Wi + Xi·Wr + bi`

### CIFAR-10 — QPSKConv2d
- Each layer maintains **two weight tensors**: `w_real` + `w_imag`
- Output: `sqrt(conv(x, w_real)² + conv(x, w_imag)² + 1e-8)` — Euclidean magnitude, always positive
- Input pipeline: `RGB (3ch) → 1×1 Float Conv (V6) / HardQAMInput (V7+) → QPSK Conv blocks → Classifier`

### Full 16-QAM ResNet
- Pre-activation ResNet blocks: `BN → QAM16Quantize → Conv3×3 → BN → QAM16Quantize → Conv3×3 → Float skip add`
- Float skip connections provide an **exact gradient highway**, solving gradient vanishing in deep quantized networks
- XNOR-Net scaling: learned per-channel float scale `α` restores dynamic range

---

## Key Techniques

### Straight-Through Estimator (STE)
Enables backprop through the non-differentiable `sign()` quantizer:
```python
class QPSKQuantize(torch.autograd.Function):
    def forward(ctx, x):
        return torch.sign(x) / math.sqrt(2)
    def backward(ctx, grad_output):
        return grad_output.clone()   # ← the STE
```

### Clipped STE (V7-Turbo)
Zeros gradients for weights that have already committed to a QPSK state:
```python
mask = (x.abs() <= 1.2).to(grad_output.dtype)
return grad_output * mask
```

### Bimodal Sign Penalty
Forces latent weights away from zero (unstable decision boundary):
```python
L_penalty = mean(exp(-|w_real|) + exp(-|w_imag|)) × λ
```

### β-Annealed Soft Quantisation (V6)
Gradually hardens quantisation from float-like (β=0.1) to rigid QPSK (β=10.0) over 100 epochs using a softmax-weighted blend over constellation points.

### Knowledge Distillation (V7-Turbo)
A float VGG teacher (88.98% acc) trains the QPSK student via KL divergence at temperature T=4.0:
```python
L_total = (1 - α)·CE(student, hard_labels) + α·KD_loss   # α = 0.7
```

---

## Training Curriculum

### MNIST — Three Phases
| Phase | Epochs | What's Quantised | Purpose |
|-------|--------|------------------|---------|
| Discovery | 1–10 | Nothing (Float32) | Learn features in continuous domain |
| Signal Adaptation | 11–18 | Activations → QAM/QPSK | Learn through discrete channels |
| Constellation Lockdown | 19–28 | Weights + Activations | Fine-tune discrete weight states |

### V7-Turbo — Full Stack
- Float VGG teacher (25 epochs) → discard at deployment
- CutMix augmentation (p=0.5) — forces structural feature learning over quantisation artifacts
- Label smoothing (ε=0.1) — prevents overconfident logits
- Cosine LR + 10-epoch warmup
- Gradient clipping (max_norm=5.0) — absorbs STE gradient spikes

---

## Ablation Study (V7 → V7-Turbo)

| Addition | Accuracy | Gain |
|----------|----------|------|
| V7 Baseline | 75.63% | — |
| + Clipped STE | ~78% | +2.4% |
| + KD (T=4.0) | ~83% | +7.4% |
| + CutMix | **86.53%** | +10.9% |

**Knowledge Distillation is the single largest contributor (+7.4%).** The training failure of QPSK networks is fundamentally an optimization landscape problem — soft teacher labels smooth the rugged gradient landscape.

---

## Diagnostic Metrics

| Metric | Formula | What it measures |
|--------|---------|------------------|
| `sign_commitment` | `(w.abs() > 0.1).float().mean()` | Fraction of weights that have committed to a QPSK state |
| `snap_error` | `MAE(w_latent, sign(w_latent)/√2)` | Distance of latent weight from its quantised snap |
| Input Quantizer Verify | `x_q.unique().numel() == expected_levels` | Sanity check before training |

---

## Hardware Tradeoff Summary

| Architecture | ADC Bits | Compression | Accuracy | Recommendation |
|---|---|---|---|---|
| V7-Turbo (16-QAM input + QPSK weights) | 1-bit | 16× | 86.53% | **Best accuracy/hardware-cost** |
| Full 16-QAM ResNet | 2-bit | 8× | 89.9% | Use if 2-bit amplitude control available |
| V8 (16-QAM weights) | 2-bit | 8× | 80.82% | Intermediate option |

---

## References

1. Liu, K. et al. *WISE: Wireless In-Situ Execution for Computing at the RF Level.* Science Advances, 2024.
2. He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
3. Hubara, I. et al. *Binarized Neural Networks.* NeurIPS, 2016.
4. Hinton, G. et al. *Distilling the Knowledge in a Neural Network.* arXiv, 2015.
5. Yun, S. et al. *CutMix: Training Strategy that Makes Strong Classifiers.* ICCV, 2019.
6. Bengio, Y. et al. *Estimating or Propagating Gradients Through Stochastic Neurons.* arXiv, 2013.

---

*Presented by: Vishnu Shah Sisodiya, Vishal Shahi, Shreyansh Bajpai, Tarang Srivastava*
