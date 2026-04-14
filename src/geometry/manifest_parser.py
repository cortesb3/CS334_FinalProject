"""
manifest_parser.py

Loads a structural manifest JSON and normalizes pixel coordinates into
world-space 3D coordinates ready for mesh generation.

Supported input formats:
  Basic   : corners [{x, y}],          lines [{x1,y1, x2,y2}]
  Enriched: corners [{x, y, z}],       lines [{x1,y1,z1, x2,y2,z2}]

Coordinate mapping (pixel → world):
  world_x =  (pixel_x / image_width  - 0.5) * world_scale
  world_y = -(pixel_y / image_height - 0.5) * world_scale   # flip: image-Y↓ → world-Y↑
  world_z =   depth_z * depth_scale                          # 0.0 when no depth present

Usage (standalone test):
  python manifest_parser.py <manifest.json>
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_manifest(json_path: str) -> dict:
    """
    Load and return the raw manifest dict from a JSON file.

    Args:
        json_path: Path to the manifest JSON.

    Returns:
        Raw dict with keys: image_width, image_height, corners, lines.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if required keys are missing.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {json_path}")

    with open(path) as f:
        data = json.load(f)

    for key in ("image_width", "image_height", "corners", "lines"):
        if key not in data:
            raise ValueError(f"Manifest missing required key: '{key}'")

    return data


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _pixel_to_world(px: float, py: float, pz: float,
                    width: int, height: int,
                    world_scale: float, depth_scale: float) -> tuple:
    """Convert one pixel-space point to world-space (x, y, z)."""
    wx =  (px / width  - 0.5) * world_scale
    wy = -(py / height - 0.5) * world_scale   # flip Y axis
    wz =   pz * depth_scale
    return (wx, wy, wz)


def normalize_to_world(manifest: dict,
                       world_scale: float = 10.0,
                       depth_scale: float = 5.0) -> dict:
    """
    Convert pixel coordinates in a manifest to world-space 3D coordinates.

    Args:
        manifest:    Raw dict from load_manifest().
        world_scale: Total world-units the image spans in X and Y (default 10 →
                     image maps to [-5, +5] on both axes).
        depth_scale: World-units that depth value 1.0 maps to on Z
                     (default 5 → depth range becomes [0, 5]).

    Returns:
        dict with:
          corners_3d : list of (x, y, z) tuples — one per corner
          lines_3d   : list of ((x1,y1,z1), (x2,y2,z2)) tuples — one per line
          world_scale: the scale used
          depth_scale: the depth scale used
    """
    w = manifest["image_width"]
    h = manifest["image_height"]

    # --- corners -------------------------------------------------------
    corners_3d = []
    for c in manifest["corners"]:
        pz = c.get("z", 0.0)   # falls back to 0 for basic (no-depth) format
        pt = _pixel_to_world(c["x"], c["y"], pz, w, h, world_scale, depth_scale)
        corners_3d.append(pt)

    # --- lines ---------------------------------------------------------
    lines_3d = []
    for seg in manifest["lines"]:
        z1 = seg.get("z1", 0.0)
        z2 = seg.get("z2", 0.0)
        p1 = _pixel_to_world(seg["x1"], seg["y1"], z1, w, h, world_scale, depth_scale)
        p2 = _pixel_to_world(seg["x2"], seg["y2"], z2, w, h, world_scale, depth_scale)
        lines_3d.append((p1, p2))

    return {
        "corners_3d": corners_3d,
        "lines_3d": lines_3d,
        "world_scale": world_scale,
        "depth_scale": depth_scale,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def parse(json_path: str,
          world_scale: float = 10.0,
          depth_scale: float = 5.0) -> dict:
    """
    One-shot helper: load + normalize.

    Returns the same dict as normalize_to_world(), plus the raw manifest
    under the key 'raw'.
    """
    raw = load_manifest(json_path)
    result = normalize_to_world(raw, world_scale=world_scale, depth_scale=depth_scale)
    result["raw"] = raw
    return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manifest_parser.py <manifest.json>")
        sys.exit(1)

    result = parse(sys.argv[1])
    corners = result["corners_3d"]
    lines   = result["lines_3d"]

    print(f"Image size : {result['raw']['image_width']} x {result['raw']['image_height']} px")
    print(f"World scale: {result['world_scale']} units  |  Depth scale: {result['depth_scale']} units")
    print(f"Corners    : {len(corners)}")
    print(f"Lines      : {len(lines)}")

    print("\nFirst 5 corners (world-space x, y, z):")
    for pt in corners[:5]:
        print(f"  ({pt[0]:+.3f}, {pt[1]:+.3f}, {pt[2]:+.3f})")

    print("\nFirst 5 lines:")
    for (p1, p2) in lines[:5]:
        print(f"  ({p1[0]:+.3f},{p1[1]:+.3f},{p1[2]:+.3f})"
              f"  →  ({p2[0]:+.3f},{p2[1]:+.3f},{p2[2]:+.3f})")

    all_pts = corners + [p for seg in lines for p in seg]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]
    print(f"\nWorld bounds:")
    print(f"  X: [{min(xs):+.3f}, {max(xs):+.3f}]")
    print(f"  Y: [{min(ys):+.3f}, {max(ys):+.3f}]")
    print(f"  Z: [{min(zs):+.3f}, {max(zs):+.3f}]")
