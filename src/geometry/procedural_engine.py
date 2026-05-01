"""
procedural_engine.py

CLI entry point: reads a structural manifest JSON and writes a 3D mesh OBJ.

Chains:
  manifest_parser  →  mesh_builder  →  obj_exporter

Output is always written to src/geometry/obj/<manifest_stem>.obj.

Usage:
  python procedural_engine.py <manifest.json> [options]

Options:
  --world-scale FLOAT   World units the image spans in X and Y  (default: 10.0)
  --depth-scale FLOAT   World units depth value 1.0 maps to     (default: 5.0)
  --beam-width  FLOAT   Width of each structural line panel      (default: 0.05)
  --no-shell            Omit the bounding-box building shell

Examples:
  python procedural_engine.py ../../depth/enriched_output.json
  python procedural_engine.py result.json --no-shell --beam-width 0.1
"""

import argparse
import sys
from pathlib import Path

# Allow running from any working directory
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

OBJ_DIR = _HERE / "obj"

from manifest_parser import parse
from mesh_builder import (build_line_beams, build_building_shell,
                           build_depth_surface, build_topology_faces,
                           build_dense_surface, build_layered_structure)
from obj_exporter import export_mesh


def run(manifest_path: str,
        output_path: str,
        world_scale: float = 10.0,
        depth_scale: float = 5.0,
        beam_width: float = 0.05,
        include_shell: bool = True,
        include_surface: bool = False,
        include_topology: bool = False,
        include_dense: bool = False,
        n_layers: int = 0) -> dict:
    """
    Full pipeline: parse manifest → build geometry → export OBJ.

    Returns the summary dict from export_mesh:
      {"vertices": int, "faces": int, "path": str}
    """
    # --- Step 1: parse ---------------------------------------------------
    result = parse(manifest_path, world_scale=world_scale, depth_scale=depth_scale)
    corners = result["corners_3d"]
    lines   = result["lines_3d"]

    print(f"Manifest : {manifest_path}")
    print(f"  Image  : {result['raw']['image_width']} x {result['raw']['image_height']} px")
    print(f"  Corners: {len(corners)}   Lines: {len(lines)}")

    # --- Step 2: build geometry ------------------------------------------
    beam_verts, beam_faces = build_line_beams(lines, beam_width=beam_width)
    print(f"  Beams  : {len(beam_verts)} vertices, {len(beam_faces)} faces")

    shell_verts, shell_faces = [], []
    if include_shell and corners:
        shell_verts, shell_faces = build_building_shell(corners)
        print(f"  Shell  : {len(shell_verts)} vertices, {len(shell_faces)} faces")

    surface_verts, surface_faces = [], []
    if include_surface and corners:
        surface_verts, surface_faces = build_depth_surface(corners)
        print(f"  Surface: {len(surface_verts)} vertices, {len(surface_faces)} faces")

    topology_verts, topology_faces = [], []
    if include_topology and (corners or lines):
        topology_verts, topology_faces = build_topology_faces(corners, lines)
        print(f"  Topology: {len(topology_verts)} vertices, {len(topology_faces)} faces")

    dense_verts, dense_faces = [], []
    if include_dense:
        depth_map_path = result["raw"].get("depth_map_path")
        if depth_map_path:
            raw = result["raw"]
            dense_verts, dense_faces = build_dense_surface(
                depth_map_path,
                raw["image_width"], raw["image_height"],
                world_scale=world_scale, depth_scale=depth_scale,
            )
            print(f"  Dense  : {len(dense_verts)} vertices, {len(dense_faces)} faces")
        else:
            print("  Dense  : skipped (no depth_map_path in manifest)")

    layer_verts, layer_faces = [], []
    if n_layers > 0 and corners:
        layer_verts, layer_faces = build_layered_structure(corners, n_layers=n_layers)
        print(f"  Layers : {len(layer_verts)} vertices, {len(layer_faces)} faces ({n_layers} bands)")

    # --- Step 3: export --------------------------------------------------
    # Merge layers into the surface slot so the exporter handles it
    all_surface_verts = surface_verts + layer_verts
    all_surface_faces = (surface_faces +
                         [(i + len(surface_verts), j + len(surface_verts), k + len(surface_verts))
                          for (i, j, k) in layer_faces])

    summary = export_mesh(shell_verts, shell_faces, beam_verts, beam_faces, output_path,
                          all_surface_verts, all_surface_faces,
                          topology_verts, topology_faces,
                          dense_verts,    dense_faces)
    print(f"Output   : {output_path}")
    print(f"  Total  : {summary['vertices']} vertices, {summary['faces']} faces")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3D OBJ mesh from a structural manifest JSON."
    )
    parser.add_argument("manifest", help="Path to manifest JSON (basic or enriched)")
    parser.add_argument("--world-scale", type=float, default=10.0,
                        help="World units the image spans in X and Y (default: 10.0)")
    parser.add_argument("--depth-scale", type=float, default=5.0,
                        help="World units depth 1.0 maps to on Z (default: 5.0)")
    parser.add_argument("--beam-width",  type=float, default=0.05,
                        help="Width of each structural line panel (default: 0.05)")
    parser.add_argument("--no-shell", action="store_true",
                        help="Omit the bounding-box building shell")
    parser.add_argument("--surface", action="store_true",
                        help="Include Delaunay depth-surface mesh from corner point cloud")
    parser.add_argument("--topology", action="store_true",
                        help="Infer wall-face polygons from the structural line graph")
    parser.add_argument("--dense", action="store_true",
                        help="Back-project the full depth map into a dense 3D surface mesh "
                             "(requires depth_map_path in the manifest)")
    parser.add_argument("--layers", type=int, default=0,
                        help="Number of horizontal sections for layered structure (e.g. 8). "
                             "Produces stepped 3D architecture from corner spread per band.")
    args = parser.parse_args()

    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OBJ_DIR / (Path(args.manifest).stem + ".obj"))

    run(
        manifest_path=args.manifest,
        output_path=output_path,
        world_scale=args.world_scale,
        depth_scale=args.depth_scale,
        beam_width=args.beam_width,
        include_shell=not args.no_shell,
        include_surface=args.surface,
        include_topology=args.topology,
        include_dense=args.dense,
        n_layers=args.layers,
    )


if __name__ == "__main__":
    main()
