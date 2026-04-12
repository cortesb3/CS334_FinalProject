"""
Visualization Script: Overlay extracted lines and corners on original image
Helps verify that feature extraction is working correctly.
"""

import cv2
import json
import sys
from pathlib import Path


def visualize_extraction(image_path, json_path, output_path):
    """
    Load an image and overlay extracted lines and corners.
    
    Args:
        image_path (str): Path to original image
        json_path (str): Path to extracted features JSON
        output_path (str): Path to save visualization
    """
    # Load the original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Load the JSON results
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Image size: {data['image_width']}x{data['image_height']}")
    print(f"Features found: {len(data['lines'])} lines, {len(data['corners'])} corners")
    
    # Draw lines in GREEN with thickness 2
    for line in data['lines']:
        x1, y1 = int(line['x1']), int(line['y1'])
        x2, y2 = int(line['x2']), int(line['y2'])
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
    
    # Draw corners as RED circles with radius 5
    for corner in data['corners']:
        x, y = int(corner['x']), int(corner['y'])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red filled circles
    
    # Save the visualization
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")
    print("\nVisualization Guide:")
    print("  - GREEN lines: Detected structural edges")
    print("  - RED circles: Detected corners/vertices")
    print("\nLook for: Lines should follow building edges, corners at key junctions")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <image_path> [json_path] [output_path]")
        print("Example: python visualize_results.py test_images/building.jpg sample_output/result.json sample_output/visualization.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else "sample_output/result.json"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "sample_output/visualization.jpg"
    
    visualize_extraction(image_path, json_path, output_path)
