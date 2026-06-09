# Evidential loss for semantic segmentation (adapted from EvSemMap/EvSemSeg),
# with the LiDAR-range-image fixes EvSemMap doesn't need (their inputs are
# dense camera images):
#   - EDL + KL averaged over VALID pixels only. Sparse range images have many
#     empty/no-return pixels (proj_mask == 0); naively dividing by H*W deflates
#     the EDL term and wastes KL pressure regularising empty-pixel Dirichlets.
#   - `ignore_index` may be a list of classes (derived from data_cfg
#     `learning_ignore`); EvSemMap only supported a single int.
#   - KL anneal window is configurable via `kl_warmup_epochs` (the EDL paper
#     uses min(1.0, epoch/10)); when unset it falls back to annealing over the
#     full max_epoch (EvSemMap-style), capped at 1.0.
#   - `unc_type` supports the paper's three loss forms: "log" / "digamma"
#     (Bayes risk under CE) and "mse" (Bayes risk under L2, Eq.5 — the form
#     the paper reports as most stable).
import torch
import torch.nn.functional as F
from math import ceil


class EvidentialLossCal:
    def __init__(self, unc_args, ignore_index=None, max_epoch=100, writer=None):
        self.unc_act = unc_args.get("unc_act", "exp")
        self.unc_type = unc_args.get("unc_type", "log")
        self.kl_strength = unc_args.get("kl_strength", 0.5)
        # KL anneal window in epochs. Paper uses min(1.0, epoch/10); null/0
        # falls back to annealing over the full max_epoch (EvSemMap-style).
        self.kl_warmup_epochs = unc_args.get("kl_warmup_epochs")
        # When False, the KL coefficient is held STATIC at kl_strength from
        # epoch 0 (no ramp). Useful for isolating noise sources, since the
        # ramp otherwise makes the training objective non-stationary.
        self.kl_anneal = unc_args.get("kl_anneal", True)
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
        elif self.unc_type == "mse":
            # Bayes-risk-under-L2 form (paper Eq.5); computed directly from
            # alpha in loss(), so no log/digamma transform is needed.
            self.unc_fn = None
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
        
        # Stashed for external logging (W&B in trainer_wb).
        self.last_edl_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_kl_coef = 0.0
        self.last_kl_weighted = 0.0
        # Dirichlet health diagnostics (populated in loss()).
        self.last_u_correct = 0.0
        self.last_u_incorrect = 0.0
        self.last_u_gap = 0.0
        self.last_evidence_mean = 0.0
        self.last_alpha_true = 0.0
        self.last_alpha_wrong = 0.0
        self.last_dead_frac = 0.0
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

        # Per-pixel evidential term. Multiplying by valid_mask zeros out
        # empty/ignored pixels before reduction.
        if self.unc_type == "mse":
            # Paper Eq.5 (Bayes risk under sum-of-squares), summed over classes:
            #   L = Σ_k (y_k - p̂_k)^2  +  Σ_k p̂_k(1 - p̂_k)/(S + 1)
            # with p̂ = alpha/S. err drives accuracy, var shrinks Dirichlet
            # spread. On invalid pixels labels_1hot is already zeroed, and the
            # valid_mask multiply below drops them entirely.
            p_hat = alpha / alpha0
            err = (labels_1hot - p_hat) ** 2
            var = p_hat * (1.0 - p_hat) / (alpha0 + 1.0)
            per_pixel = torch.sum(err + var, dim=1, keepdim=True)
        else:
            # log => Type II MLE; digamma => Bayes risk under CE. The one-hot
            # mask picks out the true-class term.
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
        if not self.kl_anneal:
            # Static: full KL weight from epoch 0 (stationary objective).
            kl_coef = self.kl_strength
        else:
            # Linear KL anneal over `kl_warmup_epochs` (paper uses 10); when
            # unset, anneal over the full run (EvSemMap-style). Clamped at
            # kl_strength so it doesn't overshoot past the warmup window.
            W = max(1, int(self.kl_warmup_epochs)) if self.kl_warmup_epochs else self.max_epoch
            kl_coef = self.kl_strength * min(1.0, max(0.0, float(curr_epoch)) / W)
        loss_kl = self._compute_kl_loss(kl_alpha, valid_mask=valid_mask)

        if self.writer is not None:
            self.writer.add_scalar("Loss/evid_kl_reg", loss_kl.item(), self.total_iter)

        self.last_edl_loss = edl_mean.item()
        self.last_kl_loss = loss_kl.item()
        self.last_kl_coef = float(kl_coef)
        # Weighted KL as it actually enters the total loss — lets us see the
        # edl_loss : kl balance directly (kl_loss alone hides the coef ramp).
        self.last_kl_weighted = float(kl_coef) * loss_kl.item()

        # --- Dirichlet health diagnostics (no grad; logged by the trainer) ---
        # These don't affect the loss; they answer "is the evidence behaving?".
        with torch.no_grad():
            K = alpha.shape[1]
            S = alpha0.squeeze(1)                       # [B,H,W]
            vm = valid_mask.squeeze(1)                  # [B,H,W] bool
            vcount = vm.sum().clamp(min=1).item()
            u = K / S.clamp(min=1e-8)                   # total uncertainty K/S
            pred = alpha.argmax(dim=1)                  # [B,H,W]
            lab = labels.squeeze(1)
            correct = (pred == lab) & vm
            incorrect = (pred != lab) & vm
            nc = correct.sum().clamp(min=1).item()
            ni = incorrect.sum().clamp(min=1).item()
            # THE core test: uncertainty should be higher on wrong predictions.
            self.last_u_correct = (u * correct).sum().item() / nc
            self.last_u_incorrect = (u * incorrect).sum().item() / ni
            self.last_u_gap = self.last_u_incorrect - self.last_u_correct
            # Evidence health: total evidence Σe_k = S - K. Collapse -> ~0
            # (flat Dirichlet, under-confident); explosion -> overconfident.
            total_evidence = S - K
            self.last_evidence_mean = (total_evidence * vm).sum().item() / vcount
            # True-class alpha should grow; wrong-class alpha should stay ~1
            # (the KL pins it there). Big gap = well-separated Dirichlet.
            alpha_true = (alpha * labels_1hot).sum(dim=1)          # [B,H,W]
            self.last_alpha_true = (alpha_true * vm).sum().item() / vcount
            alpha_wrong = (S - alpha_true) / max(1, K - 1)
            self.last_alpha_wrong = (alpha_wrong * vm).sum().item() / vcount
            # Fraction of valid pixels with no evidence at all (matters for relu).
            self.last_dead_frac = ((total_evidence < 1e-6) & vm).sum().item() / vcount

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
