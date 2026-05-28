"""Expected Calibration Error (ECE) / Max Calibration Error (MCE) meter.

Mirrors EvSemMap's `optimized_ece_with_bin` (EvSemMap/EvSemSeg/test.py) but
runs online so we can report calibration alongside IoU each validation epoch:

    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|
    MCE = max_b |acc(B_b) - conf(B_b)|        (over non-empty bins)

Confidence is expected in [0, 1]. For evidential outputs use
`certainty = 1 - K/S`; for softmax outputs use `output.max(dim=1)`.
"""

from __future__ import annotations

import torch


class EceMeter:
    def __init__(self, n_bins: int = 10, device="cpu"):
        self.n_bins = int(n_bins)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self._make_buffers()

    def _make_buffers(self):
        zeros = lambda: torch.zeros(self.n_bins, device=self.device, dtype=torch.float64)
        self.bin_count = zeros()
        self.bin_correct = zeros()
        self.bin_conf = zeros()

    def reset(self):
        self.bin_count.zero_()
        self.bin_correct.zero_()
        self.bin_conf.zero_()

    @torch.no_grad()
    def add(self, confidences, correct, valid_mask=None):
        """Accumulate one batch of per-pixel (confidence, correctness) values.

        confidences : float tensor in [0, 1], any shape.
        correct     : bool/int tensor, same shape.
        valid_mask  : optional bool tensor (same shape); only entries where
                      True are counted. Use it to drop empty-range-image
                      pixels (proj_mask == 0) and ignored classes.
        """
        c = confidences.detach().to(self.device, dtype=torch.float64).flatten()
        k = correct.detach().to(self.device, dtype=torch.float64).flatten()
        if valid_mask is not None:
            m = valid_mask.detach().to(self.device, dtype=torch.bool).flatten()
            c = c[m]
            k = k[m]
        if c.numel() == 0:
            return
        # Bin idx 0..n_bins-1; clamp so 1.0 falls in the last bin (not n_bins).
        idx = (c.clamp(0.0, 1.0 - 1e-9) * self.n_bins).long()
        ones = torch.ones_like(c)
        self.bin_count.scatter_add_(0, idx, ones)
        self.bin_correct.scatter_add_(0, idx, k)
        self.bin_conf.scatter_add_(0, idx, c)

    def compute(self):
        """Return (ece, mce) as Python floats. (0.0, 0.0) if no samples."""
        n_total = float(self.bin_count.sum())
        if n_total == 0:
            return 0.0, 0.0
        nonempty = self.bin_count > 0
        denom = self.bin_count.clamp(min=1)
        acc_per_bin = self.bin_correct / denom
        conf_per_bin = self.bin_conf / denom
        gap = (acc_per_bin - conf_per_bin).abs()
        ece = float(((self.bin_count / n_total) * gap).sum())
        mce = float(gap[nonempty].max()) if nonempty.any() else 0.0
        return ece, mce

    def n_samples(self) -> int:
        return int(self.bin_count.sum())
