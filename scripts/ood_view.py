#!/usr/bin/env python3
"""Interactive PyVista viewer for CENet OOD detection.

Shows one in-domain scan (MCD / KTH, OS1-64) next to one cross-domain scan
(KITTI-360, Velodyne HDL-64E) side by side, each point colored by its
Mahalanobis OOD score on a shared "turbo" heatmap so the two are directly
comparable. The detector and statistics are the MCD-trained ones; only the
input projection differs per sensor (handled by build_detector_and_groups).

Controls: left-drag rotate, scroll zoom, right-drag pan. The two views are
linked so they move together. Press 'q' to close.

Examples:
    python scripts/ood_view.py                 # interactive, scan index 0
    python scripts/ood_view.py --scan 10
    python scripts/ood_view.py --screenshot ood_view.png   # headless -> PNG
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ce_net import CONFIG_DIR

from ood_common import STATS_FILENAME, load_yaml
from ood_test_scans import build_detector_and_groups, score_scan

try:
    import pyvista as pv
except ImportError:
    sys.exit("PyVista not installed. Run: pip install pyvista")

GROUP_TITLES = {
    "mcd_kth": "MCD / KTH  (in-domain, OS1-64)",
    "kitti360": "KITTI-360  (cross-domain, HDL-64E)",
}


def main():
    base_cfg = load_yaml(CONFIG_DIR / "training_mcd.yaml")
    infer_cfg = load_yaml(CONFIG_DIR / "inference_mcd.yaml")
    default_model = infer_cfg.get("inference", {}).get("model_path") or os.path.join(
        base_cfg["training"]["model_path"], base_cfg["training"]["model_name"]
    )

    ap = argparse.ArgumentParser("./ood_view.py")
    ap.add_argument("--model", "-m", default=default_model)
    ap.add_argument("--datasets_root",
                    default="/media/donceykong/doncey_ssd_021/datasets")
    ap.add_argument("--mcd_seq", default="kth_day_06")
    ap.add_argument("--kitti_seq", default="2013_05_28_drive_0000_sync")
    ap.add_argument("--kitti_fov_up", type=float, default=3.0)
    ap.add_argument("--kitti_fov_down", type=float, default=-25.0)
    ap.add_argument("--scan", type=int, default=0, help="Scan index to view per group.")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--point_size", type=float, default=3.0)
    ap.add_argument("--clim", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="Fixed color range. Default: shared 1/99 percentile.")
    ap.add_argument("--pmin", type=float, default=1.0, help="Low color percentile.")
    ap.add_argument("--pmax", type=float, default=99.0, help="High color percentile.")
    ap.add_argument("--screenshot", default=None,
                    help="Render off-screen to this PNG instead of opening a window.")
    FLAGS, _ = ap.parse_known_args()

    # Need enough scans to reach the requested index.
    n = FLAGS.scan + 1
    model, hook, stats, groups, device = build_detector_and_groups(
        FLAGS.model, FLAGS.datasets_root, FLAGS.mcd_seq, FLAGS.kitti_seq, n,
        kitti_fov_up=FLAGS.kitti_fov_up, kitti_fov_down=FLAGS.kitti_fov_down,
        stats_path=FLAGS.stats,
    )

    clouds = {}
    print(f"\nScoring scan index {FLAGS.scan} per group...")
    for gname, (ds, count) in groups.items():
        if FLAGS.scan >= count:
            raise IndexError(f"--scan {FLAGS.scan} out of range for '{gname}' (has {count}).")
        xyz, scores = score_scan(model, hook, stats, ds[FLAGS.scan], device)
        clouds[gname] = (xyz, scores)
        print(f"  {gname:9s}: {len(scores)} pts, "
              f"mean={scores.mean():.2f} median={np.median(scores):.2f} max={scores.max():.2f}")
    hook.remove()

    # Shared color scale so the two heatmaps are directly comparable.
    if FLAGS.clim is not None:
        lo, hi = FLAGS.clim
    else:
        combined = np.concatenate([s for _, s in clouds.values()])
        lo, hi = np.percentile(combined, [FLAGS.pmin, FLAGS.pmax])
    print(f"Shared OOD color range: [{lo:.2f}, {hi:.2f}]")

    off_screen = FLAGS.screenshot is not None
    pl = pv.Plotter(shape=(1, 2), border=True, off_screen=off_screen,
                    title="CENet OOD: in-domain vs cross-domain")

    names = ["mcd_kth", "kitti360"]
    for col, gname in enumerate(names):
        xyz, scores = clouds[gname]
        cloud = pv.PolyData(np.ascontiguousarray(xyz, dtype=np.float32))
        cloud["OOD score"] = scores

        pl.subplot(0, col)
        pl.add_text(GROUP_TITLES.get(gname, gname), font_size=10)
        pl.add_mesh(
            cloud,
            scalars="OOD score",
            cmap="turbo",
            clim=(lo, hi),
            point_size=FLAGS.point_size,
            render_points_as_spheres=True,
            style="points",
            # One shared colorbar is enough; show it on the right panel.
            show_scalar_bar=(col == 1),
            scalar_bar_args={"title": "Mahalanobis OOD", "vertical": True},
        )
        pl.set_background("white")
        pl.add_axes()

    pl.link_views()
    pl.view_xy()

    if off_screen:
        pl.screenshot(FLAGS.screenshot)
        print(f"Saved screenshot -> {FLAGS.screenshot}")
    else:
        print("Opening viewer (press 'q' to quit)...")
        pl.show()


if __name__ == "__main__":
    main()
