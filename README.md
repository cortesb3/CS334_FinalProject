## 2D Building Structure Extraction 

### Part 1: Vision Completed
- **Loads building images** and converts to grayscale
- **Preprocesses** with aggressive Gaussian blur (13×13 kernel) to eliminate noise (siding texture, clouds, bushes)
- **Detects structural edges** using Canny edge detection with adaptive, image-aware thresholds
- **Extracts lines** via Hough Line Transform with strict parameters (minLineLength=90px, threshold=100 votes)
- **Detects corners** (vertices) using goodFeaturesToTrack with clustering post-processing
- **Exports clean JSON** with image dimensions, detected lines, and corners for downstream 3D modeling

### Key Features
✅ **Adaptive Canny thresholds** - automatically adjusts to image lighting and colors  
✅ **Aggressive noise filtering** - 89% line reduction vs. naive approach (400→45 lines on test image)  
✅ **Corner clustering** - removes duplicate detections, keeps only prominent vertices  
✅ **Clean JSON export** - standardized format with pixel coordinates  
✅ **Diagnostic output** - shows thresholds and feature counts for debugging  

### Quick Start
```bash
cd src/vision
source ../../.venv/bin/activate
python vision_extractor.py test_images/your_building.jpg output.json
python visualize_results.py test_images/your_building.jpg output.json visualization.jpg
```

### Output Structure
```json
{
  "image_width": 1400,
  "image_height": 933,
  "lines": [
    {"x1": 100, "y1": 200, "x2": 300, "y2": 200},
    ...
  ],
  "corners": [
    {"x": 150, "y": 220},
    ...
  ]
}
```

### ⚠️ Limitation
**Circular features** (like clock faces, window patterns, satellite dishes) generate significant noise in the Hough Line Transform. The algorithm is optimized for geometric buildings with flat roofs and straight lines. If your building has circular architectural elements or high-contrast repetitive textures (like decorative patterns), expect extra noise in the extracted lines. Consider:
- Pre-processing to mask circular regions
- Post-processing to merge near-parallel lines
- Increasing `threshold` parameter in HoughLinesP for stricter line selection

### Project Structure
```
src/vision/
├── vision_extractor.py        # Main extraction pipeline
├── visualize_results.py       # Visualization tool
├── test_images/               # Sample building images
└── sample_output/             # JSON results & visualizations
```