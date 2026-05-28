"""Dataset-agnostic keyframe index selection.

Given per-scan positions (world-frame xyz, in temporal order), pick a
spatially-spaced subset for training. This is the shared core; each dataset
parser supplies a `get_keyframe_scans` that reads its own pose format, builds
the positions array, and calls `select_keyframe_indices`.

Selection rule (greedy forward + relax-to-fill):
  - Walk scans in order, keeping one whenever it is >= `keyframe_dist` metres
    from the last kept keyframe; stop once `quota` are kept. Overshoot (a long
    trajectory yields more than quota) therefore keeps the first `quota` in
    order.
  - Undershoot (a slow/stationary run can't reach quota at that spacing) falls
    back to a uniform index stride across the whole sequence so the quota is
    still filled.
"""

import numpy as np


def select_keyframe_indices(positions, keyframe_dist, quota):
    """Return sorted row indices into `positions` to keep.

    Args:
        positions: (N, 3) array of world-frame xyz in temporal order.
        keyframe_dist: minimum spacing in metres between kept keyframes.
        quota: target number of indices to return.

    Returns:
        list[int] of length min(quota, N), sorted ascending.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    quota = int(quota)
    if n == 0 or quota <= 0:
        return []
    if quota >= n:
        # Asked for at least the whole sequence — take everything.
        return list(range(n))

    # Greedy forward pass: keep a scan when it's far enough from the last kept.
    kept = [0]
    last = positions[0]
    for i in range(1, n):
        if len(kept) >= quota:
            break
        if np.linalg.norm(positions[i] - last) >= keyframe_dist:
            kept.append(i)
            last = positions[i]

    if len(kept) >= quota:
        return kept[:quota]

    # Undershoot: keyframe_dist was too coarse to fill the quota. Relax to a
    # uniform stride over the whole sequence. With quota < n the rounded
    # linspace yields exactly `quota` distinct indices.
    return np.round(np.linspace(0, n - 1, quota)).astype(int).tolist()
