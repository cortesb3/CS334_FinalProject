"""
procedural_engine.py

CLI entry point: reads a structural manifest JSON and writes a 3D mesh OBJ.

Chains:
  manifest_parser  →  mesh_builder  →  obj_exporter

Usage:
  python procedural_engine.py <manifest.json> <output.obj> [options]

Options:
  --world-scale FLOAT   World units the image spans in X and Y  (default: 10.0)
  --depth-scale FLOAT   World units depth value 1.0 maps to     (default: 5.0)
  --beam-width  FLOAT   Width of each structural line panel      (default: 0.05)
  --no-shell            Omit the bounding-box building shell

Examples:
  python procedural_engine.py ../../depth/enriched_output.json building.obj
  python procedural_engine.py result.json flat.obj --no-shell --beam-width 0.1
"""

import argparse
import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from manifest_parser import parse
from mesh_builder import build_line_beams, build_building_shell
from obj_exporter import export_mesh


def run(manifest_path: str,
        output_path: str,
        world_scale: float = 10.0,
        depth_scale: float = 5.0,
        beam_width: float = 0.05,
        include_shell: bool = True) -> dict:
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

    # --- Step 3: export --------------------------------------------------
    summary = export_mesh(shell_verts, shell_faces, beam_verts, beam_faces, output_path)
    print(f"Output   : {output_path}")
    print(f"  Total  : {summary['vertices']} vertices, {summary['faces']} faces")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3D OBJ mesh from a structural manifest JSON."
    )
    parser.add_argument("manifest", help="Path to manifest JSON (basic or enriched)")
    parser.add_argument("output",   help="Path to write the output OBJ file")
    parser.add_argument("--world-scale", type=float, default=10.0,
                        help="World units the image spans in X and Y (default: 10.0)")
    parser.add_argument("--depth-scale", type=float, default=5.0,
                        help="World units depth 1.0 maps to on Z (default: 5.0)")
    parser.add_argument("--beam-width",  type=float, default=0.05,
                        help="Width of each structural line panel (default: 0.05)")
    parser.add_argument("--no-shell", action="store_true",
                        help="Omit the bounding-box building shell")
    args = parser.parse_args()

    run(
        manifest_path=args.manifest,
        output_path=args.output,
        world_scale=args.world_scale,
        depth_scale=args.depth_scale,
        beam_width=args.beam_width,
        include_shell=not args.no_shell,
    )


if __name__ == "__main__":
    main()
