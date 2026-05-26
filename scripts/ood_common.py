#!/usr/bin/env python3
"""Shared helpers for CENet Mahalanobis OOD detection.

The strategy (see strategy.png): instead of thresholding the softmax logits,
we tap the penultimate feature map produced right before CENet's final 2D
classification head and score every pixel by its Mahalanobis distance to the
nearest in-distribution (ID) class cluster.

In every CENet backbone (HarDNet / ResNet_34 / Fid) the decoder tail is:

    conv_1 (-> 256) -> conv_2 (-> 128) -> semantic_output (1x1 conv -> nclasses)

so the output of ``model.conv_2`` is the [B, 128, H, W] penultimate feature
map ``z`` from the diagram. We grab it with a forward hook (no model edits),
calibrate per-class means + a shared (tied) inverse covariance offline
(ood_calibrate.py), then score pixels online (ood_detect.py).

Reference: Lee et al., "A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018.
"""

import os

import torch
import torch.nn as nn
import yaml

# conv_2 -> 128 channels in HarDNet, ResNet_34 and Fid. The hook below asserts
# this against the live module so a future architecture change fails loudly.
FEATURE_DIM = 128

# Weights file saved inside every trained CENet run dir (see models/user.py).
WEIGHTS_FILENAME = "SENet_valid_best"

# Calibrated statistics written next to the weights by ood_calibrate.py.
STATS_FILENAME = "ood_mahalanobis.pt"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ignored_class_ids(DATA):
    """Class ids flagged ignore in data_cfg's ``learning_ignore`` (e.g. MCD 11)."""
    ignore = DATA.get("learning_ignore", {}) or {}
    return sorted(int(k) for k, v in ignore.items() if v)


def num_classes(DATA):
    """Same convention as Parser.get_n_classes(): len(learning_map_inv)."""
    return len(DATA["learning_map_inv"])


def _convert_activation(model, ARCH):
    """Replicate User's LeakyReLU->{Hardswish,SiLU} swap for res/fid pipelines.

    The activations carry no parameters, so ``load_state_dict`` would succeed
    either way -- but inference features would be wrong if the activation does
    not match what the weights were trained with. Keep this in lockstep with
    ce_net/models/user.py.
    """
    act_name = ARCH["train"].get("act")
    if act_name == "Hardswish":
        act = nn.Hardswish()
    elif act_name == "SiLU":
        act = nn.SiLU()
    else:
        return  # LeakyReLU (the module default) -- nothing to swap.

    def _swap(module):
        for child_name, child in module.named_children():
            if isinstance(child, nn.LeakyReLU):
                setattr(module, child_name, act)
            else:
                _swap(child)

    _swap(model)


def build_model(ARCH, nclasses, model_dir, device, weights_filename=WEIGHTS_FILENAME):
    """Construct a CENet backbone and load its trained weights.

    Mirrors the model-construction block of ce_net/models/user.py so the
    architecture matches the checkpoint exactly (strict load).
    """
    pipeline = ARCH["train"]["pipeline"]
    aux = ARCH["train"]["aux_loss"]

    if pipeline == "hardnet":
        from ce_net.models.network.HarDNet import HarDNet

        model = HarDNet(nclasses, aux)
    elif pipeline == "res":
        from ce_net.models.network.ResNet import ResNet_34

        model = ResNet_34(nclasses, aux)
        _convert_activation(model, ARCH)
    elif pipeline == "fid":
        from ce_net.models.network.Fid import ResNet_34

        model = ResNet_34(nclasses, aux)
        _convert_activation(model, ARCH)
    else:
        raise ValueError(f"Unknown ARCH.train.pipeline: {pipeline!r}")

    weights_path = os.path.join(model_dir, weights_filename)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    state = torch.load(weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"], strict=True)

    model.eval().to(device)
    return model


class PenultimateHook:
    """Capture the output of ``model.conv_2`` ([B, FEATURE_DIM, H, W]).

    Use as a context manager or call ``remove()`` when done. After each
    forward pass, read ``hook.features``.
    """

    def __init__(self, model):
        if not hasattr(model, "conv_2"):
            raise AttributeError(
                "Model has no 'conv_2'; cannot locate the penultimate feature "
                "map. Did the backbone change?"
            )
        self._features = None
        self._handle = model.conv_2.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if output.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                f"conv_2 produced {output.shape[1]} channels, expected "
                f"{FEATURE_DIM}. Update FEATURE_DIM in ood_common.py."
            )
        # Detach + float32: the covariance inverse and distances are sensitive,
        # so we never let AMP/half precision leak into the statistics.
        self._features = output.detach().float()

    @property
    def features(self):
        if self._features is None:
            raise RuntimeError("No features captured yet; run a forward pass first.")
        return self._features

    def remove(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def valid_pixel_mask(labels_flat, mask_flat, nclasses, ignore_ids):
    """Boolean mask over flattened pixels usable for calibration.

    Keeps pixels that (a) actually contain a projected point (proj_mask > 0),
    (b) have an in-range learning-map label, and (c) are not an ignore class.
    Note: relying on proj_mask -- not ``label > 0`` -- is what lets class 0
    ("barrier" in MCD) count as a valid ID class while empty pixels (which the
    parser also stamps with label 0 via ``proj_sem_label * proj_mask``) are
    dropped.
    """
    valid = (mask_flat > 0) & (labels_flat >= 0) & (labels_flat < nclasses)
    for ig in ignore_ids:
        valid = valid & (labels_flat != ig)
    return valid


def _is_per_class(inv_covariance):
    """A [C, C] matrix is the shared (tied) covariance; [K, C, C] is per-class."""
    return inv_covariance.dim() == 3


def mahalanobis_min_sq(feats, means, inv_covariance, class_ids):
    """Squared Mahalanobis distance to the nearest ID class, for flat features.

    Args:
        feats: [N, C] feature vectors (float).
        means: [num_classes, C] per-class means.
        inv_covariance: [C, C] tied inverse covariance, OR [num_classes, C, C]
            per-class inverse covariances (indexed by class id).
        class_ids: 1-D LongTensor of class indices to score against.
    Returns:
        (min_sq [N], argmin_class [N]) -- squared distance to the nearest class
        and which class that was.
    """
    per_class = _is_per_class(inv_covariance)
    min_sq = torch.full((feats.shape[0],), float("inf"), device=feats.device)
    argmin = torch.full((feats.shape[0],), -1, dtype=torch.long, device=feats.device)
    for c in class_ids.tolist():
        delta = feats - means[c]                                  # [N, C]
        S = inv_covariance[c] if per_class else inv_covariance    # [C, C]
        d2 = ((delta @ S) * delta).sum(dim=1)                     # [N]
        upd = d2 < min_sq
        min_sq = torch.where(upd, d2, min_sq)
        argmin = torch.where(upd, torch.full_like(argmin, c), argmin)
    return min_sq, argmin


def mahalanobis_sq_to_class(feats, means, inv_covariance, classes):
    """Squared Mahalanobis distance from each feature to its given class.

    Args:
        feats: [N, C]; classes: [N] LongTensor of a class id per point.
    Returns:
        [N] squared distances D^2_c(x) for the specified class of each point.
    """
    delta = feats - means[classes]                               # [N, C]
    if _is_per_class(inv_covariance):
        S = inv_covariance[classes]                              # [N, C, C]
        left = torch.einsum("nc,ncd->nd", delta, S)             # [N, C]
    else:
        left = delta @ inv_covariance                            # [N, C]
    return (left * delta).sum(dim=1)


def mahalanobis_scores(features, means, inv_covariance, class_ids, squared=False):
    """Per-pixel OOD score = distance to the nearest ID class.

    Args:
        features: [B, C, H, W] penultimate features.
        means: [num_classes, C] per-class means.
        inv_covariance: [C, C] tied OR [num_classes, C, C] per-class inverse
            covariance.
        class_ids: 1-D LongTensor of class indices to score against.
        squared: if True return squared distance (for the chi-square test);
            otherwise return the Mahalanobis distance.
    Returns:
        [B, H, W] score map.
    """
    B, C, H, W = features.shape
    feats = features.permute(0, 2, 3, 1).reshape(-1, C).float()  # [B*H*W, C]
    min_sq, _ = mahalanobis_min_sq(feats, means, inv_covariance, class_ids)
    out = min_sq if squared else torch.sqrt(torch.clamp(min_sq, min=0.0))
    return out.view(B, H, W)


def _norm_ppf(q):
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
               ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q > phigh:
        r = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    r = q - 0.5
    s = r * r
    return (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5]) * r / \
           (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1)


def chi2_threshold(d, q=0.975):
    """Quantile of the chi-square distribution with d dof at level q.

    Under the per-class Gaussian assumption, the squared Mahalanobis distance
    of an ID sample to its class is chi-square_d (Shojaei et al. 2024 use the
    0.975 quantile as the ID/OOD cutoff). Uses scipy if available, otherwise
    the Wilson-Hilferty approximation.
    """
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(q, d))
    except ImportError:
        import math
        z = _norm_ppf(q)
        return d * (1 - 2.0 / (9 * d) + z * math.sqrt(2.0 / (9 * d))) ** 3


def select_inv_cov(stats, mode="auto"):
    """Pick the inverse covariance from a stats dict.

    mode: "tied", "perclass", or "auto" (per-class if calibrated, else tied).
    Returns (inv_covariance, per_class_bool).
    """
    has_pc = stats.get("inv_covariances") is not None
    if mode == "perclass" or (mode == "auto" and has_pc):
        if not has_pc:
            raise ValueError("Per-class covariance requested but not in stats; "
                             "re-run ood_calibrate.py (it now saves it).")
        return stats["inv_covariances"], True
    return stats["inv_covariance"], False


def save_stats(path, means, inv_covariance, class_counts, ignore_ids, nclasses,
               inv_covariances=None):
    """Persist calibrated statistics for ood_detect.py.

    inv_covariances (optional): [nclasses, C, C] per-class inverse covariances.
    """
    class_ids = torch.nonzero(class_counts > 0, as_tuple=False).flatten().cpu()
    blob = {
        "means": means.cpu(),
        "inv_covariance": inv_covariance.cpu(),       # tied (always saved)
        "class_counts": class_counts.cpu(),
        "class_ids": class_ids,          # classes actually seen during calibration
        "ignore_ids": list(ignore_ids),
        "nclasses": int(nclasses),
        "feature_dim": FEATURE_DIM,
    }
    if inv_covariances is not None:
        blob["inv_covariances"] = inv_covariances.cpu()  # per-class
    torch.save(blob, path)


def load_stats(path, device):
    """Load calibrated statistics and move tensors onto ``device``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"OOD stats not found: {path}. Run scripts/ood_calibrate.py first."
        )
    stats = torch.load(path, map_location=device)
    stats["means"] = stats["means"].to(device).float()
    stats["inv_covariance"] = stats["inv_covariance"].to(device).float()
    stats["class_ids"] = stats["class_ids"].to(device).long()
    if stats.get("inv_covariances") is not None:
        stats["inv_covariances"] = stats["inv_covariances"].to(device).float()
    return stats
