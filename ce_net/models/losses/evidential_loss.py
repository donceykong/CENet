# Evidential loss for semantic segmentation (adapted from EvSemMap/EvSemSeg).
import torch
import torch.nn.functional as F
from math import ceil

class EvidentialLossCal:
    def __init__(self, unc_args, void_index=0, max_epoch=100, writer=None):
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
        self.total_iter = 0
        self.max_epoch = max_epoch
        self.ignore_index = void_index
        self.writer = writer
        if self.ohem is not None:
            assert 0 <= self.ohem < 1

    def logit_to_evidence(self, logit):
        return self.activation(logit)
    def evidence_to_alpha(self, evidence):
        return evidence + 1.0
    def logit_to_alpha(self, logit):
        return self.evidence_to_alpha(self.logit_to_evidence(logit))

    def expand_onehot_labels(self, label, target):
        if label.dim() == 3:
            label = label.unsqueeze(1)
        bin_labels = label.new_zeros(target.shape)
        valid_mask = (label >= 0) & (label != self.ignore_index)
        inds = torch.nonzero(valid_mask, as_tuple=True)
        if inds[0].numel() > 0:
            bin_labels[inds[0], label[valid_mask].long(), inds[2], inds[3]] = 1
        return bin_labels

    def loss(self, logits, labels, curr_iter, curr_epoch):
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        assert logits.dim() == 4 and labels.dim() == 4
        labels = labels.long()
        labels_1hot = self.expand_onehot_labels(labels, logits)
        alpha = self.logit_to_alpha(logits)
        alpha0 = torch.sum(alpha, dim=1, keepdim=True)
        edl_loss = torch.sum(labels_1hot * (self.unc_fn(alpha0) - self.unc_fn(alpha)), dim=1, keepdim=True)
        if self.ohem is not None:
            top_k = int(ceil(edl_loss.numel() * self.ohem))
            if top_k != edl_loss.numel():
                edl_loss, _ = edl_loss.topk(top_k)
        if self.writer is not None:
            self.writer.add_scalar("Loss/evid_loss", edl_loss.view(-1).mean().item(), self.total_iter)
        self.total_iter += 1
        target_c = 1.0
        kl_alpha = (alpha - target_c) * (1 - labels_1hot) + target_c
        kl_coef = self.kl_strength * (curr_epoch / max(1, self.max_epoch))
        loss_kl = self._compute_kl_loss(kl_alpha)
        if self.writer is not None:
            self.writer.add_scalar("Loss/evid_kl_reg", loss_kl.item(), self.total_iter)
        return edl_loss.view(-1).mean() + kl_coef * loss_kl

    def _dirichlet_kl_divergence(self, alphas, target_alphas):
        eps = 1e-8
        alp0 = torch.sum(alphas, dim=1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=1, keepdim=True)
        alp0_term = torch.lgamma(alp0 + eps) - torch.lgamma(target_alp0 + eps)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        alphas_term = torch.sum(torch.lgamma(target_alphas + eps) - torch.lgamma(alphas + eps) + (alphas - target_alphas) * (torch.digamma(alphas + eps) - torch.digamma(alp0 + eps)), dim=1, keepdim=True)
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        return torch.squeeze(alp0_term + alphas_term).mean()

    def _compute_kl_loss(self, alphas, target_concentration=1.0):
        target_alphas = torch.ones_like(alphas) * target_concentration
        return self._dirichlet_kl_divergence(alphas, target_alphas)
