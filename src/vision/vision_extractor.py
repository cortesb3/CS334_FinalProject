"""
Vision Extractor: 2D Building Structure Extraction

Processes 2D building images to extract structural skeleton (lines and corners)
and exports them as JSON for the Procedural Engineer to convert to 3D geometry.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path


def load_and_preprocess_image(image_path):
    """
    Load an image and preprocess it for feature extraction.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        tuple: (grayscale_image, original_image) - both as numpy arrays
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise (sky, clouds, landscaping)
    # Kernel size (5,5) is small enough to preserve edges, but large enough to smooth noise
    # Sigma ~1.0 controls the spread of the blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    return blurred, img


def extract_lines(preprocessed_image):
    """
    Extract straight lines from the preprocessed image using Canny edge detection
    and Hough Line Transform.
    
    Args:
        preprocessed_image (np.array): Grayscale, blurred image
        
    Returns:
        list: List of tuples [(x1, y1, x2, y2), ...] representing lines
    """
    # Apply Canny edge detection
    # threshold1=100, threshold2=200: detect strong vs weak edges
    # These values work well for building images with high contrast
    # Adjust lower for faint lines, higher for noisy images
    edges = cv2.Canny(preprocessed_image, 100, 200)
    
    # Use probabilistic Hough Line Transform (more efficient than standard Hough)
    # cv2.HoughLinesP returns line endpoints directly: (x1, y1, x2, y2)
    # minLineLength=50: ignore lines shorter than 50 pixels (debris, small texture)
    # maxLineGap=20: lines separated by <20 pixels are connected
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=20)
    
    # HoughLinesP returns shape (N, 1, 4), so we flatten it
    if lines is None:
        return []
    
    lines = lines.reshape(-1, 4).tolist()
    return lines


def extract_corners(preprocessed_image):
    """
    Extract corner points (structural vertices) from the preprocessed image
    using goodFeaturesToTrack.
    
    Args:
        preprocessed_image (np.array): Grayscale, blurred image
        
    Returns:
        list: List of tuples [(x, y), ...] representing corners
    """
    # goodFeaturesToTrack (Shi-Tomasi) is more robust than Harris for buildings
    # maxCorners=200: limit number of corners to avoid too much noise
    # qualityLevel=0.01: corners are 1% as good as the best corner
    # minDistance=20: minimum distance between corners (prevents clustering)
    corners = cv2.goodFeaturesToTrack(preprocessed_image, maxCorners=200,
                                      qualityLevel=0.01, minDistance=20)
    
    if corners is None:
        return []
    
    # Convert from (N, 1, 2) shape to list of tuples
    corners = corners.reshape(-1, 2).round().astype(int).tolist()
    
    return corners


def filter_features(lines, corners, image_height, image_width):
    """
    Filter and clean extracted features to prioritize prominent structural elements.
    
    Args:
        lines (list): List of [(x1, y1, x2, y2), ...] 
        corners (list): List of [(x, y), ...]
        image_height (int): Image height in pixels
        image_width (int): Image width in pixels
        
    Returns:
        tuple: (filtered_lines, filtered_corners)
    """
    filtered_lines = []
    
    # Filter lines by length: keep only lines longer than 30 pixels
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length >= 30:
            filtered_lines.append(line)
    
    # Filter corners by image bounds (remove any out-of-bounds points)
    filtered_corners = []
    for corner in corners:
        x, y = corner
        if 0 <= x < image_width and 0 <= y < image_height:
            filtered_corners.append(corner)
    
    return filtered_lines, filtered_corners


def export_to_json(lines, corners, image_width, image_height, output_path):
    """
    Export extracted features to a standardized JSON file.
    
    Args:
        lines (list): List of [(x1, y1, x2, y2), ...]
        corners (list): List of [(x, y), ...]
        image_width (int): Width of original image
        image_height (int): Height of original image
        output_path (str): Path to save JSON file
    """
    # Construct the JSON structure
    data = {
        "image_width": image_width,
        "image_height": image_height,
        "corners": [{"x": int(x), "y": int(y)} for x, y in corners],
        "lines": [{"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)} 
                  for x1, y1, x2, y2 in lines]
    }
    
    # Write to file with nice formatting
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported to: {output_path}")


def main(image_path, output_json_path):
    """
    Main pipeline: load image -> extract features -> filter -> export JSON
    
    Args:
        image_path (str): Path to input building image
        output_json_path (str): Path to save output JSON file
    """
    print(f"Processing: {image_path}")
    
    # Step 1: Preprocess
    gray, original = load_and_preprocess_image(image_path)
    height, width = original.shape[:2]
    print(f"Loaded image: {width}x{height}")
    
    # Step 2: Extract features
    lines = extract_lines(gray)
    corners = extract_corners(gray)
    print(f"Extracted: {len(lines)} lines, {len(corners)} corners (raw)")
    
    # Step 3: Filter features
    filtered_lines, filtered_corners = filter_features(lines, corners, height, width)
    print(f"Filtered: {len(filtered_lines)} lines, {len(filtered_corners)} corners")
    
    # Step 4: Export to JSON
    export_to_json(filtered_lines, filtered_corners, width, height, output_json_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vision_extractor.py <image_path> [output_json_path]")
        print("Example: python vision_extractor.py building.jpg output.json")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"
    
    main(image_path, output_json_path)
