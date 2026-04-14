"""
mesh_builder.py

Generates triangle mesh geometry from normalized 3D points and line segments
produced by manifest_parser.py.

Two geometry passes:
  build_line_beams(lines_3d, beam_width)
      Each line segment → a flat rectangular quad (4 verts, 2 triangles).
      The quad is oriented perpendicular to the line in the XY plane, giving
      a thin structural panel that mirrors the detected image edge.

  build_building_shell(corners_3d)
      Axis-aligned bounding box of all corners → a closed box mesh
      (8 verts, 12 triangles / 6 quads). Gives a solid volume for texturing.

Both functions return (vertices, faces) where:
  vertices : list of (x, y, z) float tuples
  faces    : list of (i, j, k) int tuples — 0-indexed triangle indices

Usage (standalone test):
  python mesh_builder.py
"""

import math
import sys


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(v: tuple) -> tuple:
    """Return the unit vector of v. Falls back to (1,0,0) for zero-length."""
    x, y, z = v
    mag = math.sqrt(x*x + y*y + z*z)
    if mag < 1e-9:
        return (1.0, 0.0, 0.0)
    return (x / mag, y / mag, z / mag)


def _add(a: tuple, b: tuple) -> tuple:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _sub(a: tuple, b: tuple) -> tuple:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _scale(v: tuple, s: float) -> tuple:
    return (v[0]*s, v[1]*s, v[2]*s)


def _perp_xy(d: tuple) -> tuple:
    """
    Return a unit vector perpendicular to d in the XY plane.
    If d is nearly vertical (no XY component), fall back to (1, 0, 0).
    """
    px, py = -d[1], d[0]
    mag = math.sqrt(px*px + py*py)
    if mag < 1e-9:
        return (1.0, 0.0, 0.0)
    return (px / mag, py / mag, 0.0)


# ---------------------------------------------------------------------------
# Pass A — Line beams
# ---------------------------------------------------------------------------

def build_line_beams(lines_3d: list, beam_width: float = 0.05) -> tuple:
    """
    Convert each 3D line segment into a flat rectangular quad.

    The quad lies in the plane defined by the line direction and the XY-plane
    normal, so panels face the camera for a front-facing building image.

    Args:
        lines_3d  : list of ((x1,y1,z1), (x2,y2,z2)) from manifest_parser.
        beam_width: full width of each panel in world units (default 0.05).

    Returns:
        (vertices, faces)
        vertices : list of (x, y, z) tuples
        faces    : list of (i, j, k) 0-indexed triangle tuples
    """
    half = beam_width / 2.0
    vertices = []
    faces = []

    for (p1, p2) in lines_3d:
        d = _normalize(_sub(p2, p1))
        perp = _perp_xy(d)
        offset = _scale(perp, half)

        # 4 corners of the quad:  A-  A+  B+  B-
        #   A- = p1 - offset
        #   A+ = p1 + offset
        #   B+ = p2 + offset
        #   B- = p2 - offset
        base = len(vertices)
        vertices.append(_sub(p1, offset))   # base+0  A-
        vertices.append(_add(p1, offset))   # base+1  A+
        vertices.append(_add(p2, offset))   # base+2  B+
        vertices.append(_sub(p2, offset))   # base+3  B-

        # Split quad into 2 triangles
        faces.append((base+0, base+1, base+2))
        faces.append((base+0, base+2, base+3))

    return vertices, faces


# ---------------------------------------------------------------------------
# Pass B — Building shell
# ---------------------------------------------------------------------------

def build_building_shell(corners_3d: list) -> tuple:
    """
    Build an axis-aligned bounding-box mesh from all detected corners.

    The box spans [x_min, x_max] × [y_min, y_max] × [z_min, z_max].
    When the manifest has no depth (all z=0), the box degenerates to a flat
    plane — callers may pass a min_depth to give it some thickness.

    Args:
        corners_3d: list of (x, y, z) tuples from manifest_parser.

    Returns:
        (vertices, faces)
        vertices : 8 corner vertices of the bounding box
        faces    : 12 triangles (2 per box face × 6 faces)
    """
    if not corners_3d:
        return [], []

    xs = [p[0] for p in corners_3d]
    ys = [p[1] for p in corners_3d]
    zs = [p[2] for p in corners_3d]

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    z0, z1 = min(zs), max(zs)

    # Ensure non-zero depth when all z values are the same (flat / no-depth manifest)
    if abs(z1 - z0) < 1e-6:
        z0 -= 0.5
        z1 += 0.5

    # 8 vertices of the box
    # Index layout:
    #   0: (x0,y0,z0)   1: (x1,y0,z0)   2: (x1,y1,z0)   3: (x0,y1,z0)  — front face
    #   4: (x0,y0,z1)   5: (x1,y0,z1)   6: (x1,y1,z1)   7: (x0,y1,z1)  — back face
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # 0-3 front
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # 4-7 back
    ]

    # 6 box faces, each split into 2 triangles (CCW winding, outward normals)
    faces = [
        # front  (z0)
        (0, 2, 1), (0, 3, 2),
        # back   (z1)
        (4, 5, 6), (4, 6, 7),
        # bottom (y0)
        (0, 1, 5), (0, 5, 4),
        # top    (y1)
        (3, 6, 2), (3, 7, 6),
        # left   (x0)
        (0, 4, 7), (0, 7, 3),
        # right  (x1)
        (1, 2, 6), (1, 6, 5),
    ]

    return verts, faces


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- tiny synthetic manifest so the test has zero external dependencies ---
    corners = [
        (-1.0, -1.0, 0.0),
        ( 1.0, -1.0, 2.0),
        ( 1.0,  1.0, 2.0),
        (-1.0,  1.0, 0.0),
    ]
    lines = [
        ((-1.0, -1.0, 0.0), ( 1.0, -1.0, 2.0)),
        (( 1.0, -1.0, 2.0), ( 1.0,  1.0, 2.0)),
        (( 1.0,  1.0, 2.0), (-1.0,  1.0, 0.0)),
    ]

    beam_verts, beam_faces = build_line_beams(lines, beam_width=0.1)
    print(f"Line beams : {len(lines)} lines → {len(beam_verts)} vertices, {len(beam_faces)} triangles")
    print("  First beam vertices:")
    for v in beam_verts[:4]:
        print(f"    ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")

    shell_verts, shell_faces = build_building_shell(corners)
    print(f"\nBuilding shell : {len(shell_verts)} vertices, {len(shell_faces)} triangles")
    print("  Bounding box vertices:")
    for v in shell_verts:
        print(f"    ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")

    # Sanity: every face index must be in range
    for i, f in enumerate(beam_faces):
        assert all(0 <= idx < len(beam_verts) for idx in f), f"Bad beam face {i}: {f}"
    for i, f in enumerate(shell_faces):
        assert all(0 <= idx < len(shell_verts) for idx in f), f"Bad shell face {i}: {f}"
    print("\nAll face indices valid.")
