# Quantised Neural Networks for Over-The-Air Computation
## What & Why

Neural network inference where every matrix-vector multiply happens in the **analog RF domain**. A central radio broadcasts model weights over OFDM; the edge device transmits its input; a passive RF mixer physically computes the product. No digital multiply needed.

```
Central Radio (W @ F_w) ──┐
                          RF Mixer → ADC → next layer
Edge Device    (x @ F_x) ──┘
Per subcarrier: Y[k] = W[k]·X[k]  (exact MVM in frequency domain)
```

The challenge: RF signals must be discrete constellation points (QPSK/QAM), not 32-bit floats.

---

## Architectures

**MNIST** — 784 pixels paired into 392 complex symbols → QPSK/QAM snap → Complex FC (512 → 256 → 10)

**CIFAR-10** — RGB → 1×1 float conv → 6× QPSKConv2d blocks → 2× QPSK linear → logits  
Each conv layer computes: `output = sqrt(conv(x, w_real)² + conv(x, w_imag)² + 1e-8)`

---

## Key Techniques

**Straight-Through Estimator (STE)** — `sign()` has zero gradient; STE passes gradients through unchanged during backprop so latent weights still update smoothly.

**β Annealing** — Soft quantiser blends between constellation points using softmax weights. β ramps 0.1 → 10.0 over training, transitioning from float-like to hard snap.

**Three-Phase Curriculum (MNIST)**

| Phase | Epochs | What's quantised | Purpose |
|-------|--------|-----------------|---------|
| Discovery | 1–50 | Nothing (float) | Find a good loss basin |
| Signal Adaptation | 51–100 | Activations only | Learn through discrete channels |
| Lockdown | 101–150 | Weights + activations | Fine-tune discrete states |

Skipping straight to full quantisation gives zero gradients and no learning.

**Why QPSK weights, not 16-QAM?** QPSK multiply is just a polarity flip (`±x/√2`) — trivial in analog. 16-QAM weights need a 2-bit programmable attenuator, 4× the hardware cost.

---

## Results

### MNIST
| Model | Accuracy | Compression |
|-------|----------|-------------|
| Float32 | 99% | 1× |
| 64-QAM | 97% | ~5.3× |
| QPSK | 95% | 16–32× |

### CIFAR-10
| Version | Input | Weights | Accuracy |
|---------|-------|---------|----------|
| Float baseline | Float | Float | 85.0% |
| V6 Soft QPSK | Soft QPSK | QPSK | 86.86% |
| V7 Hard 16-QAM | Hard 16-QAM | QPSK | 75.63% |
| V7-Turbo | Hard 16-QAM | QPSK | 86.53% |
| Full 16-QAM ResNet | 16-QAM | 16-QAM | **89.9%** |
| Hybrid ResNet | 16-QAM | QPSK | 88.0% |

**Notable:** V1 soft QPSK (2 bits/pixel) beats hard 64-QAM (18 bits/pixel) by 5.57% — a learned projection does semantic work that raw bandwidth can't replace.

---

## References

1. Liu et al. *WISE: Wireless In-Situ Execution.* Science Advances, 2024.
2. He et al. *Deep Residual Learning.* CVPR, 2016.
3. Hubara et al. *Binarized Neural Networks.* NeurIPS, 2016.
4. Hinton et al. *Knowledge Distillation.* arXiv, 2015.
5. Yun et al. *CutMix.* ICCV, 2019.
6. Bengio et al. *STE.* arXiv, 2013.
