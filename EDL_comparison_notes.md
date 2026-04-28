# Evidential Deep Learning (EDL) Loss: CENet vs EvSemMap Comparison

## File Locations
- **CENet**: `CENet/ce_net/models/losses/evidential_loss.py`
- **EvSemMap**: `EvSemMap/EvSemSeg/models/evidential_loss.py`

---

## 1. Core EDL Math (Identical)

Both implementations share the same core EDL pipeline:

```
logits -> evidence (activation) -> alpha (evidence + 1) -> EDL loss + KL regularizer
```

- `logit_to_evidence`: applies activation (exp/relu/softplus)
- `evidence_to_alpha`: `evidence + 1.0`
- EDL loss: `sum( labels_1hot * (unc_fn(alpha0) - unc_fn(alpha)) )` where `unc_fn` is `log` or `digamma`
- KL regularizer: Dirichlet KL divergence on non-ground-truth classes

The `_dirichlet_kl_divergence` / `dirichlet_kl_divergence` functions are equivalent (same lgamma + digamma formula, same epsilon=1e-8, same finite-check clamping).

---

## 2. Key Differences

### 2a. KL Warmup Schedule

| | CENet | EvSemMap |
|---|---|---|
| **Formula** | `kl_strength * min(1.0, epoch / 10)` | `kl_strength * (epoch / max_epoch)` |
| **Behavior** | Hard warmup over first 10 epochs, then constant | Linear ramp over entire training |

CENet has a hardcoded `W = 10` warmup window (line 69-70). The original linear schedule `kl_strength * (curr_epoch / max_epoch)` is commented out on line 68. EvSemMap ramps linearly the full duration.

**Impact**: CENet's KL regularizer reaches full strength at epoch 10 and stays there. EvSemMap's reaches full strength only at the final epoch. CENet's schedule is more aggressive and may over-regularize evidence for longer training runs.

### 2b. `compute_kl_loss` Flexibility

| | CENet | EvSemMap |
|---|---|---|
| **Signature** | `_compute_kl_loss(alphas, target_concentration=1.0)` | `compute_kl_loss(alphas, labels=None, target_concentration=1.0, concentration=1.0, reverse=True)` |
| **Features** | Forward KL only, uniform target | Supports reverse/forward KL, label-aware target, adjustable concentration |

EvSemMap's version can inject ground-truth label info into the target Dirichlet (`target_alphas += scatter(labels, target_concentration - 1)`) and can compute the KL in either direction. CENet stripped this down to just forward KL with a uniform Dirichlet(1,...,1) target.

**Impact**: For the standard EDL formulation (which both use), this doesn't matter -- both call it with uniform targets and forward KL. But EvSemMap's version would be needed for IEDL/DEDL variants.

### 2c. `evd_type` Field

EvSemMap tracks `evd_type` (`edl`, `iedl`, `dedl`) in `__init__`. CENet removed this entirely -- it only supports standard EDL.

### 2d. Loss Return Shape

| | CENet | EvSemMap |
|---|---|---|
| **Return** | `edl_loss.view(-1).mean() + kl_coef * loss_kl` (scalar) | `edl_loss + kl_coef * loss_kl` (tensor, same shape as `edl_loss`) |

CENet reduces to a scalar before returning. EvSemMap returns the per-pixel loss tensor and only reduces to scalar via `.view(-1).mean()` externally (logged on line 115 but not reduced before return on line 118).

**Impact**: EvSemMap's approach lets the caller do spatial loss weighting. CENet pre-reduces, which is fine for its trainer but less flexible.

### 2e. Extra Utilities in EvSemMap

EvSemMap has `flatten_probas()` and `mean()` helper methods that CENet removed. These are not used in the core loss computation.

### 2f. Writer Handling

EvSemMap **requires** a TensorBoard writer (calls `self.writer.add_scalar` unconditionally + `self.writer.flush()`). CENet guards all writer calls with `if self.writer is not None`, making the writer optional.

### 2g. Auxiliary Heads

| | CENet | EvSemMap |
|---|---|---|
| **Architecture** | Custom ResNet-34 / HarDNet / FidNet with optional aux heads | Stock `deeplabv3_resnet50`, single output head |
| **Aux heads** | 3 aux heads (`aux_head1/2/3`) at 1/2, 1/4, 1/8 resolution | None |
| **Aux loss in EDL mode** | Softmax + standard losses (Lovasz, CE) | N/A |

EvSemMap uses a single `deeplabv3_resnet50` encoder-decoder with one output head (`self.encoder(img)['out']`). The entire output goes through the evidential loss -- there are no auxiliary predictions.

CENet's architectures have optional auxiliary heads that produce intermediate-resolution predictions. When `aux_loss=True` and EDL is enabled, CENet runs a **hybrid training setup** (trainer.py:615-641):
- **Main head**: raw logits -> evidential loss + Lovasz on Dirichlet mean probs + boundary loss
- **Aux heads**: always apply `F.softmax()` -> standard Lovasz + CE losses

```python
# Main head (EDL)
edl_loss = self.evidential_loss_cal.loss(output, proj_labels, i, epoch)
alpha = self.evidential_loss_cal.logit_to_alpha(output)
probs_main = alpha / alpha.sum(dim=1, keepdim=True)

# Aux heads (always softmax, standard losses)
loss_m2 = 1.5 * self.ls(z2, proj_labels) + self.bd(z2, proj_labels)  # z2 already softmaxed
loss_m4 = 1.5 * self.ls(z4, proj_labels) + self.bd(z4, proj_labels)
loss_m8 = 1.5 * self.ls(z8, proj_labels) + self.bd(z8, proj_labels)
```

**Is this a problem?** Likely **no**, but it introduces a subtle training dynamic worth understanding:

1. **Shared backbone gradients come from two different loss regimes.** The main head trains the backbone via EDL gradients (which penalize overconfident wrong predictions via KL regularization on the Dirichlet). The aux heads train the backbone via standard cross-entropy/Lovasz gradients (which have no such regularization). These competing gradient signals mean the shared backbone layers receive a mix of uncertainty-aware and uncertainty-agnostic supervision.

2. **This won't corrupt inference** because only the main head is used at inference time, and the main head's own parameters (final conv layer `semantic_output`) are trained exclusively by the evidential loss path. The backbone features may be slightly different than if trained with pure EDL, but the aux losses are weighted by `lamda` (a hyperparameter) and primarily serve as intermediate supervision to help the backbone learn better multi-scale features -- a standard deep supervision technique.

3. **Potential concern for uncertainty calibration**: the backbone may learn slightly overconfident intermediate features (driven by the softmax aux heads) that the main EDL head then has to "correct." This could make the evidential head's uncertainty estimates less well-calibrated compared to EvSemMap's pure-EDL setup. If uncertainty calibration is critical, consider training with `aux_loss=False` to eliminate the competing gradient signal, or switching the aux heads to also use evidential loss.

### TODO: Incorporating EDL into the Auxiliary Heads

It is possible to make the aux heads use evidential loss too, eliminating the competing gradient signal. Two changes are required:

**Change 1: Network architectures (ResNet.py, HarDNet.py, Fid.py)** -- the aux heads hardcode `F.softmax()`. When `return_logits=True`, they should return raw logits instead. Example for ResNet.py (lines 247-255):

```python
# Current:
if self.aux:
    res_2 = self.aux_head1(res_2)
    res_2 = F.softmax(res_2, dim=1)       # <-- always softmax
    ...

# Changed:
if self.aux:
    res_2 = self.aux_head1(res_2)
    if not getattr(self, "return_logits", False):
        res_2 = F.softmax(res_2, dim=1)   # <-- skip when EDL
    ...
```

**Change 2: Trainer (trainer.py, lines 626-628)** -- treat aux outputs the same as the main head:

```python
# Current:
loss_m2 = criterion(torch.log(z2.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(z2, proj_labels.long())

# Changed:
edl_loss_2 = self.evidential_loss_cal.loss(z2, proj_labels, i, epoch)
alpha_2 = self.evidential_loss_cal.logit_to_alpha(z2)
probs_2 = alpha_2 / alpha_2.sum(dim=1, keepdim=True)
loss_m2 = edl_loss_2 + 1.5 * self.ls(probs_2, proj_labels.long())
```

Same pattern for `z4`/`z8`, and the boundary loss calls (lines 621-623) would need Dirichlet mean probs instead of the raw softmax outputs.

**Caveat**: The aux heads output at lower resolutions (1/2, 1/4, 1/8). The labels `proj_labels` may need to be downsampled to match, or the aux logits upsampled before computing EDL loss. Currently this works implicitly because NLLLoss/Lovasz handle spatial dimensions, but `expand_onehot_labels` in the evidential loss should be verified to handle mismatched spatial dims correctly.

---

## 3. Uncertainty at Inference Time

### CENet (`user.py`, lines 278-287)
```python
alpha = self.evidential_loss_cal.logit_to_alpha(proj_output)
n_classes = alpha.shape[1]
S = alpha.sum(dim=1, keepdim=True)
vacuity = (n_classes / S).squeeze(1)       # uncertainty: higher = more uncertain
probs = alpha / S                           # predictive probabilities
proj_argmax = alpha.argmax(dim=1)[0]        # predicted class
```

### EvSemMap (`unc_seg_models.py`, lines 80-97)
```python
alpha = self.logit_to_alpha(x)
num_classes = x.shape[1]
S = torch.sum(alpha, dim=1, keepdim=False)
vacuity = num_classes / S                   # uncertainty: higher = more uncertain
```

Both compute **vacuity = K/S** (num_classes / Dirichlet strength). This is correct per the EDL literature. CENet also computes `probs = alpha / S` for per-class Dirichlet mean probabilities, which is correct.

### EvSemMap's Additional Uncertainty Metrics (commented out but available)
EvSemMap has commented-out code for:
- **Differential entropy**: `sum(lgamma(alpha)) - lgamma(S) - sum((alpha-1)*(digamma(alpha) - digamma(S)))`
- **Mutual information**: `-sum((alpha/S) * (log(alpha/S) - digamma(alpha+1) + digamma(S+1)))`
- **Aleatoric uncertainty**: `1 - max_alpha / S`
- **Max-alpha certainty**: `max_alpha / S`

### CENet Inference Outputs Per Point Cloud

For each point cloud, `user.py` saves three files (lines 366-405):

| Output | Path | EDL mode | Non-EDL mode | Shape | Dtype |
|---|---|---|---|---|---|
| Predicted labels | `<label_dir>/<scan>` | argmax of alpha | argmax of softmax | `(N,)` | `int32` |
| Confidence score | `<label_dir>/confidence_scores/<scan>` | Vacuity `K/S` | Max softmax prob | `(N,)` | `float16` |
| Per-class probs | `<label_dir>/multiclass_confidence_scores/<scan>` | Dirichlet mean `alpha/S` | Softmax probs | `(N, C)` | `float16` |

**WARNING -- Semantic flip in `confidence_scores`**: The meaning of the confidence score is **inverted** between EDL and non-EDL mode. In EDL mode, higher values mean **more uncertain** (vacuity approaches 1 when evidence is low). In non-EDL mode, higher values mean **more confident** (max softmax prob approaches 1 when the model is sure). Any downstream code consuming `confidence_scores` must account for this depending on which mode produced the file.

---

## 4. Potential Issues / Concerns with CENet's Implementation

### 4a. Correctness of Vacuity Computation -- LOOKS CORRECT
The vacuity formula `K/S` is standard EDL (Sensoy et al., 2018). When the model has no evidence (all evidence = 0), alpha = 1 for all classes, S = K, and vacuity = 1 (maximum uncertainty). As evidence grows, S >> K and vacuity -> 0. This is correct.

### 4b. `model.return_logits` Flag -- VERIFIED CORRECT
CENet sets `self.model.return_logits = True` when using evidential inference (user.py:133, trainer.py:198). All three network architectures properly respect this flag:

- **ResNet.py** (line 242-245): `if getattr(self, "return_logits", False): out = logits` else `F.softmax(logits, dim=1)`
- **HarDNet.py** (line 308-311): same pattern
- **Fid.py** (line 180-183): same pattern

When `return_logits=True`, raw logits are returned (no softmax applied), so the evidential computation operates on the correct values.

**Note on auxiliary heads**: The aux heads (`aux_head1/2/3`) always apply softmax regardless of `return_logits`. This is intentional -- only the main head is trained with evidential loss. The aux heads use standard cross-entropy/Lovasz losses during training and are not used during inference.

### 4c. Prediction from Alpha vs Logits
CENet uses `alpha.argmax(dim=1)` for predictions. Since alpha = activation(logit) + 1 and all common activations (exp, relu, softplus) are monotonic, `argmax(alpha) == argmax(logit)`. This is correct.

### 4d. Training Loss Combination
In CENet's trainer (trainer.py:644-649), the EDL loss is combined with Lovasz softmax and boundary loss:
```python
edl_loss = self.evidential_loss_cal.loss(output, proj_labels, i, epoch)
alpha = self.evidential_loss_cal.logit_to_alpha(output)
probs = alpha / alpha.sum(dim=1, keepdim=True)
loss_m = edl_loss + 1.5 * lovasz(probs, labels) + boundary_loss(probs, labels)
```
The Lovasz and boundary losses receive `probs = alpha/S` (Dirichlet mean), not softmax output. This is reasonable but differs from standard practice where those losses get softmax probabilities. The Dirichlet mean probabilities sum to 1 and are valid probability distributions, so this should work, but behavior may differ slightly from what Lovasz softmax was designed for.

### 4e. Validation Path
During validation (trainer.py:810-812):
```python
alpha = self.evidential_loss_cal.logit_to_alpha(output)
output = alpha / alpha.sum(dim=1, keepdim=True)  # replaces output with probs
log_out = torch.log(output.clamp(min=1e-8))       # then takes log for NLL
```
This feeds Dirichlet mean probabilities into NLLLoss for validation metrics. This is fine for accuracy/IoU evaluation.

---

## 5. Summary: What to Verify Before Trusting CENet's Uncertainty

1. **`return_logits` is verified correct** -- all three architectures (ResNet, HarDNet, Fid) properly skip softmax when this flag is set. The uncertainty computation receives raw logits as intended.

2. **The KL warmup schedule differs** from EvSemMap (10-epoch hard warmup vs full-training linear ramp). This affects how aggressively non-ground-truth evidence is suppressed during training, which directly impacts uncertainty calibration.

3. **The core math is correct** -- vacuity = K/S from alpha = activation(logits) + 1 follows the standard EDL formulation.

4. **Consider adding other uncertainty metrics** from EvSemMap (differential entropy, mutual information) if vacuity alone doesn't provide good calibration for your use case.
