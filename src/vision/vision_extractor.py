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
    
    # Blur image to remove texture noise (siding, clouds, bushes)
    # Kernel (13, 13) smooths details while preserving main structural edges
    # Sigma 2.0 controls blur amount
    blurred = cv2.GaussianBlur(gray, (13, 13), 2.0)
    
    return blurred, img


def calculate_adaptive_canny_thresholds(blurred_image):
    """
    Calculate Canny edge detection thresholds based on image intensity statistics.
    This makes the algorithm robust to different lighting conditions and building colors.
    
    Args:
        blurred_image (np.array): Grayscale blurred image
        
    Returns:
        tuple: (threshold1, threshold2) - two thresholds for cv2.Canny
    """
    # Calculate thresholds from median pixel intensity (avoids hardcoding)
    median_intensity = np.median(blurred_image)
    sigma = 0.33
    
    # Scale: lower ~0.66x median, upper ~1.33x median
    lower_threshold = max(0, int((1.0 - sigma) * median_intensity))
    upper_threshold = min(255, int((1.0 + sigma) * median_intensity))
    
    # Make sure lower < upper
    lower_threshold = max(30, lower_threshold)
    upper_threshold = max(lower_threshold + 1, upper_threshold)
    
    return lower_threshold, upper_threshold


def extract_lines(preprocessed_image):
    """
    Extract straight lines from the preprocessed image using Canny edge detection
    and Hough Line Transform with AGGRESSIVE noise filtering.
    
    Args:
        preprocessed_image (np.array): Grayscale, blurred image
        
    Returns:
        list: List of tuples [(x1, y1, x2, y2), ...] representing lines
    """
    # Get adaptive Canny thresholds
    lower_thresh, upper_thresh = calculate_adaptive_canny_thresholds(preprocessed_image)
    edges = cv2.Canny(preprocessed_image, lower_thresh, upper_thresh)
    
    # threshold=100: need 100+ votes to count as a line (filters small noise)
    # minLineLength=90: ignore short fragments (siding texture, etc)
    # maxLineGap=40: connect broken lines (shadows can break edges)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=90, maxLineGap=40)
    
    # Reshape from (N,1,4) to list
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
    # maxCorners=150: get good corners only (not too many)
    # qualityLevel=0.02: only accept quality corners (2% of best)
    # minDistance=25: space out corners to avoid clustering
    corners = cv2.goodFeaturesToTrack(preprocessed_image, maxCorners=150,
                                      qualityLevel=0.02, minDistance=25)
    
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
