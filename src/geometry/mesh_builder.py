"""
mesh_builder.py

Generates triangle mesh geometry from normalized 3D points and line segments
produced by manifest_parser.py.

Four geometry passes:
  build_line_beams(lines_3d, beam_width, beam_depth)
      Each line segment → a rectangular prism (8 verts, 12 triangles).

  build_building_shell(corners_3d)
      Axis-aligned bounding box of all corners → a closed box mesh
      (8 verts, 12 triangles / 6 quads). Gives a solid volume for texturing.

  build_depth_surface(corners_3d, max_edge_length)
      Delaunay triangulation of the corner point cloud in XY, using each
      corner's actual Z depth. Smooth 2.5D surface (good for overview).

  build_layered_structure(corners_3d, n_layers, thickness)
      Divides corners into horizontal Y-bands and extrudes each as a solid
      box. Each band's width comes from the actual corner spread at that
      height and its Z position from the average depth of corners in that
      band. Produces genuine stepped 3D architecture (spire → belfry →
      clock section → base) rather than a smooth ramp.

Both functions return (vertices, faces) where:
  vertices : list of (x, y, z) float tuples
  faces    : list of (i, j, k) int tuples — 0-indexed triangle indices

Usage (standalone test):
  python mesh_builder.py
"""

import math
import sys
import numpy as np
from scipy.spatial import Delaunay


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

def build_line_beams(lines_3d: list, beam_width: float = 0.05, beam_depth: float = 0.1) -> tuple:
    """
    Convert each 3D line segment into a rectangular prism (box beam).

    Each beam has 8 vertices and 6 faces (12 triangles):
      - Width: offset perpendicular to the line in the XY plane
      - Depth: offset along the Z axis (into/out of the facade)

    Args:
        lines_3d  : list of ((x1,y1,z1), (x2,y2,z2)) from manifest_parser.
        beam_width: full width of the beam cross-section (default 0.05).
        beam_depth: full depth of the beam cross-section along Z (default 0.1).

    Returns:
        (vertices, faces)
        vertices : list of (x, y, z) tuples
        faces    : list of (i, j, k) 0-indexed triangle tuples
    """
    half_w = beam_width / 2.0
    half_d = beam_depth / 2.0
    vertices = []
    faces = []

    for (p1, p2) in lines_3d:
        d = _normalize(_sub(p2, p1))
        perp = _perp_xy(d)
        w = _scale(perp, half_w)   # width offset (XY plane)
        dz = (0.0, 0.0, half_d)   # depth offset (Z axis)

        # 8 vertices — front face (z + half_d) then back face (z - half_d)
        # Layout:
        #   0: p1 - w + dz    1: p1 + w + dz
        #   2: p2 + w + dz    3: p2 - w + dz   <- front face
        #   4: p1 - w - dz    5: p1 + w - dz
        #   6: p2 + w - dz    7: p2 - w - dz   <- back face
        base = len(vertices)
        vertices.append(_add(_sub(p1, w), dz))  # 0
        vertices.append(_add(_add(p1, w), dz))  # 1
        vertices.append(_add(_add(p2, w), dz))  # 2
        vertices.append(_add(_sub(p2, w), dz))  # 3
        vertices.append(_sub(_sub(p1, w), dz))  # 4
        vertices.append(_sub(_add(p1, w), dz))  # 5
        vertices.append(_sub(_add(p2, w), dz))  # 6
        vertices.append(_sub(_sub(p2, w), dz))  # 7

        b = base
        faces += [
            (b+0, b+1, b+2), (b+0, b+2, b+3),  # front
            (b+4, b+6, b+5), (b+4, b+7, b+6),  # back
            (b+0, b+4, b+5), (b+0, b+5, b+1),  # left
            (b+2, b+6, b+7), (b+2, b+7, b+3),  # right
            (b+0, b+3, b+7), (b+0, b+7, b+4),  # start cap
            (b+1, b+5, b+6), (b+1, b+6, b+2),  # end cap
        ]

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
# Pass C — Depth surface (Delaunay triangulation of corner point cloud)
# ---------------------------------------------------------------------------

def build_depth_surface(corners_3d: list, max_edge_length: float = None) -> tuple:
    """
    Build a 3D surface mesh by Delaunay-triangulating the corner point cloud.

    Each corner's (x, y) position is used for triangulation; its z value
    (from the depth map) positions it in 3D space. The result is a surface
    that follows the real depth profile of the building rather than a uniform
    bounding box.

    Triangles where any edge exceeds max_edge_length are removed — this
    filters out large spanning triangles that connect unrelated parts of the
    structure across empty space.

    Args:
        corners_3d      : list of (x, y, z) tuples from manifest_parser.
        max_edge_length : maximum allowed edge length in world units.
                          Defaults to 20% of the XY bounding-box diagonal.

    Returns:
        (vertices, faces) — vertices are the original corners_3d points;
        faces are (i, j, k) 0-indexed triangle tuples.
    """
    if len(corners_3d) < 3:
        return [], []

    pts = np.array(corners_3d, dtype=float)  # shape (N, 3)
    xy  = pts[:, :2]

    tri = Delaunay(xy)

    # Auto-compute max_edge_length from the XY bounding box if not provided
    if max_edge_length is None:
        x_range = xy[:, 0].max() - xy[:, 0].min()
        y_range = xy[:, 1].max() - xy[:, 1].min()
        diag = math.sqrt(x_range**2 + y_range**2)
        max_edge_length = diag * 0.20   # keep triangles within 20% of diagonal

    faces = []
    for simplex in tri.simplices:
        i, j, k = simplex
        # Measure edge lengths in full 3D (includes Z depth difference)
        p, q, r = pts[i], pts[j], pts[k]
        e0 = np.linalg.norm(p - q)
        e1 = np.linalg.norm(q - r)
        e2 = np.linalg.norm(r - p)
        if max(e0, e1, e2) <= max_edge_length:
            faces.append((int(i), int(j), int(k)))

    return corners_3d, faces


# ---------------------------------------------------------------------------
# Pass D — Topology inference (planar face finding)
# ---------------------------------------------------------------------------

def _snap_to_unified(raw_points: list, snap_dist: float):
    """
    Merge nearby points into a unified vertex list.

    For each incoming (x, y, z) point, if it lands within snap_dist of an
    already-unified vertex (measured in XY only), it maps to that vertex.
    Otherwise it becomes a new unified vertex.

    Returns:
        unified  : list of (x, y, z) — the de-duplicated vertex positions
        mapping  : list of int — mapping[i] = unified index for raw_points[i]
    """
    unified = []
    mapping = []
    for px, py, pz in raw_points:
        best = None
        best_d = snap_dist
        for idx, (ux, uy, _uz) in enumerate(unified):
            d = math.sqrt((px - ux) ** 2 + (py - uy) ** 2)
            if d < best_d:
                best_d = d
                best = idx
        if best is None:
            mapping.append(len(unified))
            unified.append((px, py, pz))
        else:
            mapping.append(best)
    return unified, mapping


def _signed_area_2d(pts):
    """Signed area of a 2D polygon (positive = CCW, negative = CW)."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _find_planar_faces(unified: list, adj: dict) -> list:
    """
    Find all interior faces of a planar graph using half-edge traversal.

    For each vertex, neighbours are sorted CCW by polar angle. For a directed
    half-edge (u→v), the next half-edge in the same face is (v→w) where w is
    the neighbour just BEFORE u in v's CCW list — i.e. we turn clockwise as
    sharply as possible. This traces interior faces CCW and the outer face CW.

    Returns:
        List of faces; each face is a list of vertex indices (unified list).
        The outer (unbounded) face and degenerate faces are excluded.
    """
    # Sort each vertex's neighbours by angle
    sorted_nbrs = {}
    for v, nbrs in adj.items():
        vx, vy = unified[v][0], unified[v][1]
        sorted_nbrs[v] = sorted(
            nbrs,
            key=lambda u: math.atan2(unified[u][1] - vy, unified[u][0] - vx)
        )

    def next_half_edge(u, v):
        nbrs = sorted_nbrs[v]
        idx  = nbrs.index(u)
        w    = nbrs[(idx - 1) % len(nbrs)]
        return (v, w)

    visited = set()
    faces   = []

    for u in list(adj.keys()):
        for v in adj[u]:
            if (u, v) in visited:
                continue
            face  = []
            curr  = (u, v)
            limit = len(unified) + 2   # safety cap
            while curr not in visited and len(face) <= limit:
                visited.add(curr)
                face.append(curr[0])
                curr = next_half_edge(curr[0], curr[1])

            if len(face) < 3 or len(face) > 8:
                continue
            area = _signed_area_2d([unified[i] for i in face])
            if area > 1e-6:          # keep only CCW (interior) faces
                faces.append(face)

    return faces


def build_topology_faces(corners_3d: list,
                         lines_3d: list,
                         snap_distance: float = 0.15) -> tuple:
    """
    Infer wall-face polygons from the structural line graph.

    Steps:
      1. Snap all line endpoints + corner points into a unified vertex set.
      2. Build an undirected adjacency list from the (snapped) line edges.
      3. Find interior faces using planar half-edge traversal.
      4. Fan-triangulate each polygon face.

    Args:
        corners_3d    : list of (x, y, z) from manifest_parser.
        lines_3d      : list of ((x1,y1,z1),(x2,y2,z2)) from manifest_parser.
        snap_distance : XY distance within which two points are merged (world units).

    Returns:
        (vertices, faces)
        vertices : unified vertex list (x, y, z)
        faces    : list of (i, j, k) 0-indexed triangle tuples
    """
    if not lines_3d:
        return [], []

    # --- Step 1: unify all points ----------------------------------------
    raw_pts = []
    for p1, p2 in lines_3d:
        raw_pts.append(p1)
        raw_pts.append(p2)
    for c in corners_3d:
        raw_pts.append(c)

    unified, mapping = _snap_to_unified(raw_pts, snap_distance)

    # --- Step 2: adjacency from line edges --------------------------------
    adj = {i: [] for i in range(len(unified))}
    n_line_pts = len(lines_3d) * 2
    for k in range(0, n_line_pts, 2):
        u = mapping[k]
        v = mapping[k + 1]
        if u != v and v not in adj[u]:
            adj[u].append(v)
        if u != v and u not in adj[v]:
            adj[v].append(u)

    # Remove isolated vertices (no edges — typically unconnected corners)
    adj = {v: nbrs for v, nbrs in adj.items() if nbrs}

    # --- Step 3: find planar faces ----------------------------------------
    poly_faces = _find_planar_faces(unified, adj)

    # --- Step 4: fan triangulate each polygon -----------------------------
    tri_faces = []
    for poly in poly_faces:
        anchor = poly[0]
        for i in range(1, len(poly) - 1):
            tri_faces.append((anchor, poly[i], poly[i + 1]))

    return unified, tri_faces


# ---------------------------------------------------------------------------
# Pass E — Dense depth back-projection
# ---------------------------------------------------------------------------

def build_dense_surface(depth_map_path: str,
                        image_width: int,
                        image_height: int,
                        world_scale: float = 10.0,
                        depth_scale: float = 5.0,
                        step: int = 15) -> tuple:
    """
    Build a dense mesh by back-projecting the full depth map into 3D.

    Samples every `step` pixels from the depth map, converts each sample to
    world-space using the same formula as manifest_parser, then connects
    adjacent samples into triangles to form a continuous depth surface.

    Args:
        depth_map_path: Path to a .npy file produced by pipeline.py (H×W float).
        image_width:    Original image width in pixels.
        image_height:   Original image height in pixels.
        world_scale:    World units the image spans in X and Y (match manifest).
        depth_scale:    World units depth 1.0 maps to on Z (match manifest).
        step:           Pixel stride for sampling (default 15).

    Returns:
        (vertices, faces)
    """
    depth_map = np.load(depth_map_path).astype(float)  # shape (H, W)

    row_indices = list(range(0, image_height, step))
    col_indices = list(range(0, image_width,  step))
    n_rows = len(row_indices)
    n_cols = len(col_indices)

    vertices  = []
    grid_idx  = {}   # (ri, ci) → vertex index

    for ri, py in enumerate(row_indices):
        for ci, px in enumerate(col_indices):
            py_c = min(py, depth_map.shape[0] - 1)
            px_c = min(px, depth_map.shape[1] - 1)
            pz   = float(depth_map[py_c, px_c])

            wx =  (px / image_width  - 0.5) * world_scale
            wy = -(py / image_height - 0.5) * world_scale   # flip Y
            wz =   pz * depth_scale

            grid_idx[(ri, ci)] = len(vertices)
            vertices.append((wx, wy, wz))

    faces = []
    for ri in range(n_rows - 1):
        for ci in range(n_cols - 1):
            tl = grid_idx[(ri,     ci)]
            tr = grid_idx[(ri,     ci + 1)]
            bl = grid_idx[(ri + 1, ci)]
            br = grid_idx[(ri + 1, ci + 1)]
            faces.append((tl, bl, tr))
            faces.append((tr, bl, br))

    return vertices, faces


# ---------------------------------------------------------------------------
# Pass F — Layered structural sections
# ---------------------------------------------------------------------------

def build_layered_structure(corners_3d: list,
                            n_layers: int = 8,
                            thickness: float = 0.4) -> tuple:
    """
    Divide corners into horizontal Y-bands and extrude each as a solid box.

    For each band:
      - X extents come from the min/max X of corners in that band, giving
        each section the correct width (narrow spire, wide belfry, etc.).
      - Z position comes from the average depth of corners in that band, so
        sections that are physically closer/farther sit at the right depth.
      - thickness sets how deep each section is along Z.

    The result is a stack of properly-proportioned 3D boxes that match the
    building's silhouette and depth profile — genuine architectural sections
    rather than a smooth ramp.

    Args:
        corners_3d : list of (x, y, z) tuples from manifest_parser.
        n_layers   : number of horizontal bands (default 8).
        thickness  : Z depth of each section in world units (default 0.4).

    Returns:
        (vertices, faces)
    """
    if not corners_3d:
        return [], []

    pts = np.array(corners_3d, dtype=float)
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    band_h = (y_max - y_min) / n_layers

    vertices = []
    faces = []
    half_t = thickness / 2.0

    for i in range(n_layers):
        y0 = y_min + i * band_h
        y1 = y0 + band_h

        # Corners in this Y band (include the upper boundary in the top band)
        mask = (pts[:, 1] >= y0) & (pts[:, 1] <= y1 if i == n_layers - 1 else pts[:, 1] < y1)
        band = pts[mask]

        if len(band) < 2:
            continue

        x0, x1  = band[:, 0].min(), band[:, 0].max()
        z_center = band[:, 2].mean()
        z_front  = z_center + half_t
        z_back   = z_center - half_t

        # 8 vertices of this box section
        # front face (z_front): 0-3, back face (z_back): 4-7
        base = len(vertices)
        vertices.extend([
            (x0, y0, z_front), (x1, y0, z_front),   # 0 1  bottom-front
            (x1, y1, z_front), (x0, y1, z_front),   # 2 3  top-front
            (x0, y0, z_back),  (x1, y0, z_back),    # 4 5  bottom-back
            (x1, y1, z_back),  (x0, y1, z_back),    # 6 7  top-back
        ])

        b = base
        faces += [
            (b+0, b+1, b+2), (b+0, b+2, b+3),   # front
            (b+4, b+6, b+5), (b+4, b+7, b+6),   # back
            (b+0, b+4, b+7), (b+0, b+7, b+3),   # left
            (b+1, b+5, b+6), (b+1, b+6, b+2),   # right
            (b+0, b+1, b+5), (b+0, b+5, b+4),   # bottom
            (b+3, b+2, b+6), (b+3, b+6, b+7),   # top
        ]

    return vertices, faces


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

    beam_verts, beam_faces = build_line_beams(lines, beam_width=0.1, beam_depth=0.2)
    print(f"Line beams : {len(lines)} lines → {len(beam_verts)} vertices, {len(beam_faces)} triangles")
    print(f"  Expected : {len(lines)*8} vertices, {len(lines)*12} triangles")
    print("  First beam vertices (8):")
    for v in beam_verts[:8]:
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
