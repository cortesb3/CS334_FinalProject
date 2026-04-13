import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for Anaconda + PyTorch conflict on Windows

"""
Full Pipeline: Image -> Vision Extraction -> Depth Enrichment -> Enriched JSON

This script chains together:
  1. Bea's vision_extractor.py  (2D lines + corners from image)
  2. Emma's enrich_manifest.py  (appends Z-depth to every point)

Usage:
    python pipeline.py <image_path> <output_json_path>

Example:
    python pipeline.py ../vision/test_images/Purdue_Bell_Tower.jpg enriched_output.json
"""

import sys
import json
from pathlib import Path

# --- Import Bea's vision module ---
# Assumes this script lives in src/depth/ and Bea's code is in src/vision/
vision_path = Path(__file__).parent.parent / "vision"
sys.path.insert(0, str(vision_path))
from vision_extractor import load_and_preprocess_image, extract_lines, extract_corners, filter_features, export_to_json

# --- Import Emma's depth module ---
from depth_estimator import estimate_depth, get_depth_at_point


def run_pipeline(image_path: str, output_path: str) -> dict:
    """
    Runs the full vision + depth pipeline on a single image.

    Args:
        image_path:  Path to the input building image
        output_path: Path to save the final enriched JSON

    Returns:
        enriched manifest as a Python dict
    """

    print("=" * 50)
    print("STEP 1: Vision Extraction (Bea's module)")
    print("=" * 50)

    # --- Bea's pipeline ---
    gray, original = load_and_preprocess_image(image_path)
    height, width = original.shape[:2]
    print(f"Loaded image: {width}x{height}")

    lines = extract_lines(gray)
    corners = extract_corners(gray)
    print(f"Raw extracted: {len(lines)} lines, {len(corners)} corners")

    filtered_lines, filtered_corners = filter_features(lines, corners, height, width)
    print(f"Filtered: {len(filtered_lines)} lines, {len(filtered_corners)} corners")

    # Build the intermediate manifest (same format as Bea's JSON output)
    manifest = {
        "image_width": width,
        "image_height": height,
        "corners": [{"x": int(x), "y": int(y)} for x, y in filtered_corners],
        "lines": [{"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                  for x1, y1, x2, y2 in filtered_lines]
    }

    print("\n" + "=" * 50)
    print("STEP 2: Depth Enrichment (Emma's module)")
    print("=" * 50)

    # --- Emma's pipeline ---
    depth_map = estimate_depth(image_path)

    # Enrich corners with Z
    enriched_corners = []
    for corner in manifest["corners"]:
        x, y = corner["x"], corner["y"]
        z = get_depth_at_point(depth_map, x, y)
        enriched_corners.append({"x": x, "y": y, "z": round(z, 6)})

    # Enrich line endpoints with Z
    enriched_lines = []
    for line in manifest["lines"]:
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        z1 = get_depth_at_point(depth_map, x1, y1)
        z2 = get_depth_at_point(depth_map, x2, y2)
        enriched_lines.append({
            "x1": x1, "y1": y1, "z1": round(z1, 6),
            "x2": x2, "y2": y2, "z2": round(z2, 6),
        })

    print(f"Enriched {len(enriched_corners)} corners with Z-coordinates.")
    print(f"Enriched {len(enriched_lines)} lines with Z-coordinates.")

    # --- Build final enriched manifest ---
    enriched = {
        "image_width": width,
        "image_height": height,
        "depth_source": image_path,
        "corners": enriched_corners,
        "lines": enriched_lines,
    }

    # --- Save to disk ---
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Pipeline complete! Output saved to: {output_path}")
    print("=" * 50)

    # Preview
    print("\n--- Preview ---")
    print(f"  Corners (first 3): {enriched['corners'][:3]}")
    print(f"  Lines   (first 3): {enriched['lines'][:3]}")

    return enriched


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <image_path> <output_json_path>")
        print("Example: python pipeline.py ../vision/test_images/Purdue_Bell_Tower.jpg enriched_output.json")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2]

    run_pipeline(image_path, output_path)