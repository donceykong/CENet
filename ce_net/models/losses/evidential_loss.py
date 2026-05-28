# Evidential loss for semantic segmentation (adapted from EvSemMap/EvSemSeg),
# with the LiDAR-range-image fixes EvSemMap doesn't need (their inputs are
# dense camera images):
#   - EDL + KL averaged over VALID pixels only. Sparse range images have many
#     empty/no-return pixels (proj_mask == 0); naively dividing by H*W deflates
#     the EDL term and wastes KL pressure regularising empty-pixel Dirichlets.
#   - `ignore_index` may be a list of classes (derived from data_cfg
#     `learning_ignore`); EvSemMap only supported a single int.
#   - Annealing schedule matches EvSemMap: `kl_strength * curr_epoch /
#     max_epoch` (capped at 1.0 in case training exceeds max_epoch).
import torch
import torch.nn.functional as F
from math import ceil


class EvidentialLossCal:
    def __init__(self, unc_args, ignore_index=None, max_epoch=100, writer=None):
        self.unc_act = unc_args.get("unc_act", "exp")
        self.unc_type = unc_args.get("unc_type", "log")
        self.kl_strength = unc_args.get("kl_strength", 0.5)
        self.ohem = unc_args.get("ohem")
        if self.unc_act == "exp":
            self.activation = lambda x: torch.exp(torch.clamp(x, -10, 10))
        elif self.unc_act == "relu":
            self.activation = torch.relu
        elif self.unc_act == "softplus":
            self.activation = F.softplus
        else:
            raise NotImplementedError(self.unc_act)
        if self.unc_type == "digamma":
            self.unc_fn = torch.digamma
        elif self.unc_type == "log":
            self.unc_fn = torch.log
        else:
            raise NotImplementedError(self.unc_type)

        # Normalise ignore_index to a list of int classes.
        if ignore_index is None:
            self.ignore_classes = []
        elif isinstance(ignore_index, (int,)):
            self.ignore_classes = [int(ignore_index)]
        else:
            self.ignore_classes = sorted({int(c) for c in ignore_index})

        self.total_iter = 0
        self.max_epoch = max(1, int(max_epoch))
        self.writer = writer
        
        # Stashed for external logging (Aim in trainer_wb).
        self.last_edl_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_kl_coef = 0.0
        if self.ohem is not None:
            assert 0 <= self.ohem < 1

    # --- evidence / Dirichlet helpers --------------------------------------
    def logit_to_evidence(self, logit):
        return self.activation(logit)

    def evidence_to_alpha(self, evidence):
        return evidence + 1.0

    def logit_to_alpha(self, logit):
        return self.evidence_to_alpha(self.logit_to_evidence(logit))

    # --- masking ------------------------------------------------------------
    def _build_valid_mask(self, labels, proj_mask):
        """[B,1,H,W] bool: True where the pixel both has a LiDAR return
        (proj_mask) and its label isn't in the ignore set."""
        if proj_mask is not None:
            # `.to(labels.device)` keeps this correct when the caller forgot
            # to move proj_mask to CUDA (e.g. the multi-GPU val branch).
            proj_mask = proj_mask.to(labels.device)
            if proj_mask.dim() == 3:
                proj_mask = proj_mask.unsqueeze(1)
            valid = proj_mask.bool()
        else:
            valid = torch.ones_like(labels, dtype=torch.bool)
        valid = valid & (labels >= 0)
        for c in self.ignore_classes:
            valid = valid & (labels != c)
        return valid

    def _expand_onehot(self, labels, target_shape, valid_mask):
        """[B,C,H,W] one-hot, zeroed where valid_mask is False."""
        _, C, _, _ = target_shape
        # Clip labels so unexpected values (e.g. 65535 from uint16 unlabeled)
        # don't index past the LUT; invalid pixels are masked out below.
        safe = labels.clamp(0, C - 1).long().squeeze(1)  # [B,H,W]
        bin_labels = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).to(dtype=torch.float32)
        return bin_labels * valid_mask.to(dtype=bin_labels.dtype)

    # Back-compat shim: external callers that still pass (label, target_tensor)
    # and don't have proj_mask. Uses (label in ignore set) as best-effort mask.
    def expand_onehot_labels(self, label, target):
        if label.dim() == 3:
            label = label.unsqueeze(1)
        valid = (label >= 0)
        for c in self.ignore_classes:
            valid = valid & (label != c)
        return self._expand_onehot(label, target.shape, valid)

    # --- loss ---------------------------------------------------------------
    def loss(self, logits, labels, proj_mask=None, curr_iter=0, curr_epoch=0):
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        assert logits.dim() == 4 and labels.dim() == 4
        labels = labels.long()

        valid_mask = self._build_valid_mask(labels, proj_mask)         # [B,1,H,W]
        valid_count = valid_mask.sum().clamp(min=1)

        labels_1hot = self._expand_onehot(labels, logits.shape, valid_mask)
        alpha = self.logit_to_alpha(logits)
        alpha0 = torch.sum(alpha, dim=1, keepdim=True)

        # Per-pixel evidential CE term (log or digamma form). Multiplying by
        # valid_mask zeros out empty/ignored pixels before reduction.
        per_pixel = torch.sum(
            labels_1hot * (self.unc_fn(alpha0) - self.unc_fn(alpha)),
            dim=1, keepdim=True,
        )
        per_pixel = per_pixel * valid_mask.to(per_pixel.dtype)

        if self.ohem is not None:
            valid_vals = per_pixel[valid_mask]
            if valid_vals.numel() > 0:
                top_k = int(ceil(valid_vals.numel() * self.ohem))
                if 0 < top_k < valid_vals.numel():
                    valid_vals, _ = valid_vals.topk(top_k)
                edl_mean = valid_vals.mean()
            else:
                edl_mean = per_pixel.sum() * 0  # zero with grad lineage
        else:
            edl_mean = per_pixel.sum() / valid_count

        if self.writer is not None:
            self.writer.add_scalar("Loss/evid_loss", edl_mean.item(), self.total_iter)
        self.total_iter += 1

        # KL regulariser toward a uniform Dirichlet on the WRONG classes only
        # (kl_alpha keeps alpha=1 on the true class so it isn't penalised).
        target_c = 1.0
        kl_alpha = (alpha - target_c) * (1 - labels_1hot) + target_c
        # EvSemMap-style linear anneal, clamped at kl_strength so it doesn't
        # overshoot if a stage runs past max_epoch.
        kl_coef = self.kl_strength * min(1.0, max(0.0, float(curr_epoch)) / self.max_epoch)
        loss_kl = self._compute_kl_loss(kl_alpha, valid_mask=valid_mask)

        if self.writer is not None:
            self.writer.add_scalar("Loss/evid_kl_reg", loss_kl.item(), self.total_iter)

        self.last_edl_loss = edl_mean.item()
        self.last_kl_loss = loss_kl.item()
        self.last_kl_coef = float(kl_coef)

        return edl_mean + kl_coef * loss_kl

    # --- Dirichlet KL -------------------------------------------------------
    def _dirichlet_kl_divergence(self, alphas, target_alphas, valid_mask=None):
        eps = 1e-8
        alp0 = torch.sum(alphas, dim=1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=1, keepdim=True)
        alp0_term = torch.lgamma(alp0 + eps) - torch.lgamma(target_alp0 + eps)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        alphas_term = torch.sum(
            torch.lgamma(target_alphas + eps) - torch.lgamma(alphas + eps)
            + (alphas - target_alphas)
            * (torch.digamma(alphas + eps) - torch.digamma(alp0 + eps)),
            dim=1, keepdim=True,
        )
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        per_pixel = (alp0_term + alphas_term).squeeze(1)  # [B,H,W]
        if valid_mask is not None:
            m = valid_mask.squeeze(1) if valid_mask.dim() == 4 else valid_mask
            denom = m.sum().clamp(min=1)
            return (per_pixel * m.to(per_pixel.dtype)).sum() / denom
        return per_pixel.mean()

    def _compute_kl_loss(self, alphas, target_concentration=1.0, valid_mask=None):
        target_alphas = torch.ones_like(alphas) * target_concentration
        return self._dirichlet_kl_divergence(alphas, target_alphas, valid_mask=valid_mask)
