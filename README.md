<div align="center">

<img src="assets/banner.svg" alt="Saranyan M" width="760"/>

</div>

<img src="assets/divider.svg" width="100%" alt=""/>

## `$ cat ./stack`

<div align="center">
<table>
<tr>
<td align="center" width="110"><img src="https://cdn.simpleicons.org/python/3776AB" width="44" height="44" alt="Python"/><br/><sub><b>Python</b></sub></td>
<td align="center" width="110"><img src="https://cdn.simpleicons.org/cplusplus/00599C" width="44" height="44" alt="C++"/><br/><sub><b>C / C++</b></sub></td>
<td align="center" width="110"><img src="https://cdn.simpleicons.org/pytorch/EE4C2C" width="44" height="44" alt="PyTorch"/><br/><sub><b>PyTorch</b></sub></td>
<td align="center" width="110"><img src="https://cdn.simpleicons.org/tensorflow/FF6F00" width="44" height="44" alt="TensorFlow"/><br/><sub><b>TensorFlow</b></sub></td>
<td align="center" width="110"><img src="https://cdn.simpleicons.org/keras/D00000" width="44" height="44" alt="Keras"/><br/><sub><b>Keras</b></sub></td>
<td align="center" width="110"><img src="https://cdn.simpleicons.org/onnx/005CED" width="44" height="44" alt="ONNX Runtime"/><br/><sub><b>ONNX Runtime</b></sub></td>
<td align="center" width="110"><img src="https://cdn.simpleicons.org/opencv/5C3EE8" width="44" height="44" alt="OpenCV"/><br/><sub><b>OpenCV</b></sub></td>
</tr>
</table>
</div>

## `$ cat ./experience`

<div align="center">
<table>
<tr>
<td align="center" width="130"><h2>🧠</h2><sub><b>Deep Learning</b></sub></td>
<td align="center" width="130"><h2>🤖</h2><sub><b>Transformer<br/>Architectures</b></sub></td>
<td align="center" width="130"><h2>💬</h2><sub><b>NLP</b></sub></td>
<td align="center" width="130"><h2>👁️</h2><sub><b>Computer<br/>Vision</b></sub></td>
</tr>
<tr>
<td align="center" width="130"><h2>♾️</h2><sub><b>Continual<br/>Learning</b></sub></td>
<td align="center" width="130"><h2>⚙️</h2><sub><b>Custom Optimizer<br/>Design</b></sub></td>
<td align="center" width="130"><h2>🧪</h2><sub><b>Research<br/>Prototyping</b></sub></td>
<td align="center" width="130"><h2>📝</h2><sub><b>Scientific<br/>Writing</b></sub></td>
</tr>
</table>
</div>

<img src="assets/divider.svg" width="100%" alt=""/>

## `$ ls ./projects`

```
 #   PROJECT                                      DOMAIN
 ─────────────────────────────────────────────────────────────────────
 01  PivotAdam                                    Memory-Efficient Optimization
 02  Continual Few-Shot Learning                  Continual Learning
 03  Spectral Filter Utility Modulation           Continual Learning
 04  Bilinear Fallacy Detection & Steering        Interpretability / NLP
```

<img src="assets/divider.svg" width="100%" alt=""/>

## 01 · PivotAdam: Memory-Efficient Full-Parameter Optimization

> A low-rank optimizer that makes full-parameter fine-tuning of billion-parameter models fit on 8GB consumer GPUs.

```
  gradients ──► Randomized QR ──► orthonormal subspace P (PᵀP = I)
                                        │
  momentum m, v ──► T = Pᵀ·P_new ───────┴──► pivoted into the new basis
                    (computed in float32, then cast back to fp16)
```

**The problem:** AdamW stores two extra float32 states (`m`, `v`) per parameter. On a 1.5B model that is ~18.5 GB, which is out of memory on an 8GB GPU.

**Results** (Qwen-2.5-1.5B, rank 128, 8GB GPU)

| Metric        | Flora      | GaLore     | **PivotAdam** |
|---------------|------------|------------|---------------|
| Peak VRAM     | ~9,800 MB  | ~7,700 MB  | **~7,500 MB** |
| Throughput    | 0.43 it/s  | 0.37 it/s  | **1.10 it/s** |
| Training loss | Comparable | Comparable | Comparable    |

AdamW needs ~18,500 MB on the same model. PivotAdam matches Flora and GaLore on loss, runs about 3x faster than GaLore, and uses the least VRAM of all methods.

**Core ideas**
- **Randomized QR projection:** replaces the SVD used by GaLore and Flora with an orthonormal basis at O(Nd) cost, cutting per-step overhead
- **Momentum pivoting:** when the basis refreshes, a transition matrix `T = PᵀP_new` carries the existing `m` and `v` buffers into the new coordinates, preserving the training trajectory
- **Numerical stability fix:** subspace transitions run in float32 before casting back to half precision, preventing silent underflow of the variance state

**What the ablations show**
- Replacing the projected gradient with noise stalls the loss (~9.2 vs ~3.77), so the projection carries a real descent signal
- Resetting momentum performs comparably to full transport, so `T` acts as a correctness guarantee rather than a speed driver
- Robust across rotation frequencies (loss stays within ~3.77 to 3.89); rank 128 is the recommended default
- Built for fine-tuning: when training GPT-2 Small from scratch, AdamW still wins

```python
from pivot_adam import PivotAdam

optimizer = PivotAdam(model.parameters(), lr=1e-4, rank=128, rotate_every=200)
```

`Randomized QR` `Low-Rank Optimizer` `PyTorch` `LLM Fine-Tuning`

<img src="assets/divider.svg" width="100%" alt=""/>

## 02 · Continual Few-Shot Learning

> A CV framework for lifelong adaptation from minimal examples.

```
  new class ──► few examples ──► adapt ──► old knowledge retained
                                  (no catastrophic forgetting)
```

- Generates novel classes and adapts continually over time
- Learns and generalizes from minimal training examples (few-shot)
- Avoids catastrophic forgetting

`Few-Shot Learning` `Continual Learning` `Computer Vision`

<img src="assets/divider.svg" width="100%" alt=""/>

## 03 · Spectral Filter Utility Modulation for Continual Learning

> One-shot continual learning that protects the filters a network still needs.

```
  task t ──► Spectral Filter Flux Tracking ──► utility scoring
                                                    │
        scaled LR per filter group ◄── overlay modulator
```

- One-shot continual learning framework across **20 CIFAR-100 tasks** (task-incremental)
- **Spectral Filter Flux Tracking** and activation-gradient utility scoring to detect and quantify disruption of critical filters
- Overlay modulator dynamically scales learning rates for specific filter groups, restricting updates to previously useful filters

`Continual Learning` `CIFAR-100` `Catastrophic Forgetting`

<img src="assets/divider.svg" width="100%" alt=""/>

## 04 · Bilinear Fallacy Detection & Steering (BFDS)

> Inference-time detection and correction of logical fallacies in small AI models, with no retraining.

```
  premises ──► layer probes ──► vectors (u, v) ──► bilinear score (u × v)
                                                         │
                          steer model  ◄── threshold ◄───┘
                          (1–10% frequency cap)
```

- Internal layer probes map interacting premises into vectors **u** and **v**
- Bilinear activation score (u × v) identifies logical errors in the data stream
- Activation threshold tuned against normal text, limiting steering to a **1–10%** frequency to prevent overcorrection

`Interpretability` `Activation Steering` `NLP`

<br>

<div align="center">

<img src="assets/footer.svg" alt="training loop" width="760"/>

</div>
