import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for Anaconda + PyTorch conflict on Windows

import json
import argparse
import numpy as np

from depth_estimator import estimate_depth, get_depth_at_point


def enrich_manifest(manifest_path: str, image_path: str, output_path: str) -> dict:
    """
    Reads Bea's structural manifest JSON and appends Z-coordinates to every
    corner and line endpoint using MiDaS depth estimation.

    Args:
        manifest_path: Path to Bea's output JSON (structural manifest)
        image_path:    Path to the original building image (same one Bea used)
        output_path:   Path to save the enriched JSON

    Returns:
        enriched: The enriched manifest as a Python dict
    """

    # --- Load Bea's manifest ---
    print(f"Loading manifest from: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # --- Run depth estimation on the same image ---
    print(f"Running depth estimation on: {image_path}")
    depth_map = estimate_depth(image_path)

    # --- Sanity check: make sure depth map matches manifest image dimensions ---
    depth_h, depth_w = depth_map.shape
    manifest_w = manifest.get("image_width")
    manifest_h = manifest.get("image_height")

    if manifest_w and manifest_h:
        if depth_w != manifest_w or depth_h != manifest_h:
            print(
                f"WARNING: Depth map size ({depth_w}x{depth_h}) does not match "
                f"manifest size ({manifest_w}x{manifest_h}). "
                f"Z-values may be slightly off."
            )

    # --- Enrich corners with Z ---
    enriched_corners = []
    for corner in manifest.get("corners", []):
        x, y = corner["x"], corner["y"]
        z = get_depth_at_point(depth_map, x, y)
        enriched_corners.append({"x": x, "y": y, "z": round(z, 6)})

    print(f"Enriched {len(enriched_corners)} corners with Z-coordinates.")

    # --- Enrich line endpoints with Z ---
    enriched_lines = []
    for line in manifest.get("lines", []):
        x1, y1 = line["x1"], line["y1"]
        x2, y2 = line["x2"], line["y2"]
        z1 = get_depth_at_point(depth_map, x1, y1)
        z2 = get_depth_at_point(depth_map, x2, y2)
        enriched_lines.append({
            "x1": x1, "y1": y1, "z1": round(z1, 6),
            "x2": x2, "y2": y2, "z2": round(z2, 6),
        })

    print(f"Enriched {len(enriched_lines)} lines with Z-coordinates.")

    # --- Build enriched manifest ---
    enriched = {
        "image_width": manifest.get("image_width"),
        "image_height": manifest.get("image_height"),
        "depth_source": image_path,
        "corners": enriched_corners,
        "lines": enriched_lines,
    }

    # --- Save to disk ---
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"Enriched manifest saved to: {output_path}")
    return enriched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich a structural manifest JSON with Z-coordinates from MiDaS depth estimation."
    )
    parser.add_argument("manifest", help="Path to Bea's structural manifest JSON")
    parser.add_argument("image",    help="Path to the original building image")
    parser.add_argument("output",   help="Path to save the enriched JSON output")
    args = parser.parse_args()

    result = enrich_manifest(args.manifest, args.image, args.output)

    # Print a small preview
    print("\n--- Preview of enriched manifest ---")
    preview = {
        "image_width":  result["image_width"],
        "image_height": result["image_height"],
        "corners (first 3)": result["corners"][:3],
        "lines (first 3)":   result["lines"][:3],
    }
    print(json.dumps(preview, indent=2))