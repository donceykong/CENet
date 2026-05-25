"""Sensor-config loading helpers.

A sensor yaml in `ce_net/config/SENSORS/<name>.yaml` carries the physical
properties of a LiDAR (fov, name, type). The range-image dimensions
(`img_prop.width` / `img_prop.height`) are supplied by the model config so
all sensors in a training run project to a single, batchable shape.
"""

from pathlib import Path
from typing import Mapping, Union

import yaml

from ce_net import CONFIG_DIR


SENSORS_DIR = CONFIG_DIR / "SENSORS"


def _resolve_sensor_path(name_or_path: Union[str, Path]) -> Path:
    """Accept either a bare name ("OS1_64") or a path to a sensor yaml."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        return p
    candidate = SENSORS_DIR / f"{name_or_path}.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Sensor config not found: tried {candidate}. "
            f"Pass a name from {SENSORS_DIR} (without .yaml) or a full path."
        )
    return candidate


def load_sensor(
    name_or_path: Union[str, Path],
    img_width: int,
    img_height: int,
) -> dict:
    """Load a SENSORS yaml and stamp the range-image dimensions on it.

    Returns a dict with the keys consumed by LaserScan / parser code:
        name, type, fov_up, fov_down, img_prop: {width, height}
    """
    path = _resolve_sensor_path(name_or_path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    sensor = raw.get("sensor")
    if sensor is None:
        raise ValueError(f"{path} is missing a top-level 'sensor:' block.")
    sensor.setdefault("img_prop", {})
    sensor["img_prop"]["width"] = int(img_width)
    sensor["img_prop"]["height"] = int(img_height)
    return sensor


def materialize_sensor_groups(
    sensor_groups,
    img_width: int,
    img_height: int,
):
    """Resolve every group's `sensor` field to a fully-formed sensor dict.

    Input:  [{"sensor": "OS1_64", "sequences": [...]}, ...]
    Output: [{"sensor_name": "OS1_64", "sensor": {...full dict...}, "sequences": [...]}, ...]
    """
    if not isinstance(sensor_groups, list):
        raise ValueError(
            "data_cfg.sensor_groups must be a list of {sensor, sequences} entries."
        )
    out = []
    for grp in sensor_groups:
        if not isinstance(grp, Mapping) or "sensor" not in grp or "sequences" not in grp:
            raise ValueError(
                "Every sensor_groups entry must have 'sensor' and 'sequences' keys."
            )
        sensor_name = grp["sensor"]
        out.append(
            {
                "sensor_name": sensor_name,
                "sensor": load_sensor(sensor_name, img_width, img_height),
                "sequences": list(grp["sequences"]),
            }
        )
    return out
