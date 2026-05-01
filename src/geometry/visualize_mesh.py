"""
visualize_mesh.py

Renders a generated OBJ file as an interactive 3D wireframe using matplotlib.
Shell (bounding box) and beams (structural line panels) are drawn in different
colors. The plot window is rotatable by click-dragging.

Usage:
  python visualize_mesh.py <path_to_obj>

Example:
  python visualize_mesh.py obj/result_bell_tower.obj
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# OBJ reader
# ---------------------------------------------------------------------------

def load_obj(obj_path: str) -> dict:
    """
    Parse a minimal OBJ file into vertices and named face groups.

    Returns:
        {
          "vertices": [(x, y, z), ...],
          "groups":   {"shell": [(i,j,k), ...], "beams": [...], ...}
        }
    """
    vertices = []
    groups = {}
    current_group = "default"

    for line in Path(obj_path).read_text().splitlines():
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:])
            vertices.append((x, y, z))
        elif line.startswith("g "):
            current_group = line.split()[1]
            groups.setdefault(current_group, [])
        elif line.startswith("f "):
            indices = [int(tok) - 1 for tok in line.split()[1:]]  # 0-indexed
            groups.setdefault(current_group, []).append(tuple(indices))

    return {"vertices": vertices, "groups": groups}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def visualize(obj_path: str) -> None:
    data = load_obj(obj_path)
    verts = data["vertices"]
    groups = data["groups"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    style = {
        "shell":    {"facecolors": (0.4, 0.6, 1.0, 0.10), "edgecolors": (0.2, 0.4, 0.9, 0.4), "linewidths": 0.6},
        "beams":    {"facecolors": (1.0, 0.5, 0.2, 0.6),  "edgecolors": (0.8, 0.3, 0.0, 0.9), "linewidths": 0.6},
        "surface":  {"facecolors": (0.5, 0.9, 0.5, 0.4),  "edgecolors": (0.1, 0.6, 0.1, 0.7), "linewidths": 0.4},
        "topology": {"facecolors": (0.9, 0.8, 0.2, 0.5),  "edgecolors": (0.7, 0.5, 0.0, 0.8), "linewidths": 0.5},
        "dense":    {"facecolors": (0.7, 0.4, 0.9, 0.25), "edgecolors": (0.5, 0.2, 0.7, 0.4), "linewidths": 0.3},
        "default":  {"facecolors": (0.6, 0.6, 0.6, 0.3),  "edgecolors": (0.3, 0.3, 0.3, 0.7), "linewidths": 0.6},
    }

    for group_name, faces in groups.items():
        if not faces:
            continue
        triangles = [[verts[i] for i in face] for face in faces]
        s = style.get(group_name, style["default"])
        poly = Poly3DCollection(triangles, **s)
        ax.add_collection3d(poly)

    # Set axis limits from vertex bounds
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(min(zs), max(zs))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (depth)")

    stem = Path(obj_path).stem
    ax.set_title(f"{stem}  —  {len(verts)} vertices, {sum(len(f) for f in groups.values())} faces")

    # Legend
    from matplotlib.patches import Patch
    legend_items = []
    if "shell" in groups:
        legend_items.append(Patch(facecolor=(0.4, 0.6, 1.0), edgecolor=(0.2, 0.4, 0.9), label="shell"))
    if "beams" in groups:
        legend_items.append(Patch(facecolor=(1.0, 0.5, 0.2), edgecolor=(0.8, 0.3, 0.0), label="beams"))
    if "surface" in groups:
        legend_items.append(Patch(facecolor=(0.5, 0.9, 0.5), edgecolor=(0.1, 0.6, 0.1), label="surface"))
    if "topology" in groups:
        legend_items.append(Patch(facecolor=(0.9, 0.8, 0.2), edgecolor=(0.7, 0.5, 0.0), label="topology"))
    if "dense" in groups:
        legend_items.append(Patch(facecolor=(0.7, 0.4, 0.9), edgecolor=(0.5, 0.2, 0.7), label="dense"))
    if legend_items:
        ax.legend(handles=legend_items, loc="upper left")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_mesh.py <path_to_obj>")
        print("Example: python visualize_mesh.py obj/result_bell_tower.obj")
        sys.exit(1)

    visualize(sys.argv[1])
