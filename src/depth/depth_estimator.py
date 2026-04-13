import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for Anaconda + PyTorch conflict on Windows

import cv2
import torch
import numpy as np
import argparse


def load_midas_model():
    """Download and load the MiDaS model from torch.hub."""
    print("Loading MiDaS model (this may take a moment on first run to download)...")

    model_type = "MiDaS_small"  # Fastest, good enough for our use case
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    # Load the matching transforms for this model
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    return model, transform


def estimate_depth(image_path: str, output_path: str = None) -> np.ndarray:
    """
    Run MiDaS depth estimation on a single image.

    Args:
        image_path: Path to the input image (jpg, png, etc.)
        output_path: Optional path to save the depth map as a grayscale image.
                     If None, nothing is saved to disk.

    Returns:
        depth_map: A 2D numpy array (H x W) of normalized depth values in [0, 1].
                   Higher values = closer to camera (we invert MiDaS output for intuition).
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load model
    model, transform = load_midas_model()

    # Preprocess
    input_tensor = transform(img_rgb)

    # Run inference
    print("Running depth estimation...")
    with torch.no_grad():
        prediction = model(input_tensor)

        # Resize output back to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_raw = prediction.cpu().numpy()

    # Normalize to [0, 1]
    d_min, d_max = depth_raw.min(), depth_raw.max()
    depth_normalized = (depth_raw - d_min) / (d_max - d_min + 1e-8)

    # Invert: MiDaS outputs higher = farther, we want higher = closer
    depth_map = 1.0 - depth_normalized

    # Save visualization if requested
    if output_path:
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(f"src/depth/{output_path}", depth_colored)
        print(f"Depth map saved to: {output_path}")

    print(f"Depth estimation complete. Shape: {depth_map.shape}, Range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    return depth_map


def get_depth_at_point(depth_map: np.ndarray, x: int, y: int) -> float:
    """
    Sample the depth map at a specific pixel coordinate.
    Clamps to image bounds automatically.

    Args:
        depth_map: 2D numpy array from estimate_depth()
        x: pixel x coordinate (column)
        y: pixel y coordinate (row)

    Returns:
        Normalized depth value in [0, 1]
    """
    h, w = depth_map.shape
    x_clamped = max(0, min(x, w - 1))
    y_clamped = max(0, min(y, h - 1))
    return float(depth_map[y_clamped, x_clamped])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiDaS Monocular Depth Estimator")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the depth map visualization (e.g. depth.jpg)"
    )
    args = parser.parse_args()

    depth = estimate_depth(args.image, output_path=args.output)

    # Quick test: sample depth at center of image
    h, w = depth.shape
    center_depth = get_depth_at_point(depth, w // 2, h // 2)
    print(f"Depth at image center ({w//2}, {h//2}): {center_depth:.4f}")