# Vision Guided Procedural Modeling

**Team Members:** Albert Wu, Bea Cortes, Eli Dubizh, Emma Tarrence

---

## Background and Motivation

Creating a 3D model of a real world building typically requires either a fleet of cameras, expensive LiDAR scanners, or many hours of manual work in tools like Blender. None of these are realistic for someone who just wants a quick 3D approximation from a photo they took on their phone.

We wanted to build a pipeline that takes a single photograph as input and produces a 3D structural skeleton of the building in it. This is a hard problem because a photo throws away depth information. The camera flattens a 3D scene onto a 2D sensor, and recovering that lost dimension requires either multiple viewpoints or an intelligent guess. Our project combines classical computer vision for 2D feature extraction with an AI depth model to fill in the missing Z axis, keeping the whole system accessible and controllable.

---

## Related Work / Prior Art

Three main approaches exist for turning 2D images into 3D geometry.

**Monocular depth estimation** uses a single image to predict how far away each pixel is. Intel's MiDaS model (Ranftl et al., 2020) is the most widely used tool for this. It outputs a relative depth map, meaning it tells you what is closer or farther but not exact real world distances.

**Generative AI based 3D generation** feeds a 2D image into a neural network and gets a 3D mesh back directly. These work well on small isolated objects but are essentially a black box and struggle with large architectural structures.

**Procedural and semantic reconstruction**, used in tools like CityEngine by Esri, identifies meaningful parts of a building like walls and roof edges and uses mathematical rules to build geometry. These systems are accurate but require a lot of labeled training data or manual setup.

Our project is closest to the third approach, but we skip the labeled data requirement entirely by using classical computer vision for feature detection and AI only for depth.

---

## Technical Approach

Our system is a three stage pipeline where each module passes data to the next through a JSON file.

### Stage 1: Vision Pipeline

The vision module (`src/vision/vision_extractor.py`) takes a building photo and extracts its structural skeleton as detected lines and corners.

First, the image is converted to grayscale and blurred with a 13x13 Gaussian kernel. This step is important because it suppresses fine textures like brick or glass that would otherwise flood the edge detector with noise. Next, we run adaptive Canny edge detection, computing thresholds dynamically from the image's median intensity so the algorithm works across different lighting conditions without manual tuning. We then run OpenCV's probabilistic Hough Line Transform with a vote threshold of 100 and a minimum line length of 90 pixels, which filters out short spurious detections. Finally we find structural corners using `goodFeaturesToTrack` and remove any duplicates within 30 pixels of each other.

On our main test image of the Purdue Bell Tower, this reduced roughly 400 raw detected edges down to about 45 clean structural lines and 60 corners.

**Output:** A JSON file with image dimensions, line segments as pixel coordinates, and corners as pixel x and y positions.

### Stage 2: Depth Estimation

The depth module (`src/depth/depth_estimator.py`) takes the vision JSON and the original image, runs a neural depth model, and adds a Z coordinate to every detected feature.

We use Intel's MiDaS Small model loaded through PyTorch Hub. It produces a depth map in arbitrary units that we normalize to a 0 to 1 range and then invert so that 1.0 means closest to the camera. We then sample the depth map at each corner and line endpoint pixel location to get the corresponding Z value.

**Output:** An enriched JSON where every corner is `{x, y, z}` and every line endpoint has a Z value attached.

### Stage 3: Geometry and Coordinate Normalization

The geometry module (`src/geometry/manifest_parser.py`) converts pixel space coordinates into world space 3D coordinates using the following transformation:

```
world_x =  (pixel_x / image_width  - 0.5) * world_scale
world_y = -(pixel_y / image_height - 0.5) * world_scale
world_z =   depth_z * depth_scale
```

The Y axis is flipped because image space Y increases downward while 3D world space Y increases upward. The default world scale is 10.0 and depth scale is 5.0, mapping the image to a reasonable world space range. Mesh generation from these coordinates is the next step and is currently in progress.

---

## Timeline and Development Process

Albert and Bea built the vision pipeline first. Early versions used fixed Canny thresholds and returned hundreds of noisy lines. Multiple rounds of parameter tuning brought the output to a usable level. The Purdue Bell Tower was the main test case throughout because it has both easy geometry (flat walls) and hard geometry (clock faces, arches).

Emma then integrated the MiDaS depth model to add Z coordinates to the vision output and produced false color depth visualizations to verify the results looked correct.

Eli wrote the manifest parser to normalize coordinates to world space, and the full three stage pipeline was wired together in `src/depth/pipeline.py`. Mesh generation and rendering remain as the final outstanding milestone.

---

## Current Results

The vision pipeline reliably extracts building edges from photographs. The depth module produces plausible depth maps where foreground elements correctly show higher depth values than background sky. The manifest parser correctly converts all coordinates to world space, verified by checking that the resulting point cloud falls within expected bounds.

The pipeline does not yet produce renderable 3D geometry. Output currently ends at a structured list of 3D line segments and corner points in world space. Below are example visualizations from the Purdue Bell Tower test: the left image shows detected lines in green and corners as red circles overlaid on the original photo, and the right shows the false color depth map output.

*(Insert visualization images here before submission)*

---

## Video and Code Links

**Codebase:** [https://github.com/cortesb3/CS334_FinalProject](https://github.com/cortesb3/CS334_FinalProject)

**Demo Video:** *(add link before submission)*

---

## Limitations and Future Work

The biggest limitation right now is that mesh generation is unfinished, so there is no actual 3D model output yet. The vision pipeline also struggles with circular features like clock faces because the Hough Line Transform only works for straight lines. The MiDaS model gives relative depth rather than real world distances, so the scale of the output is not physically accurate.

To make this a stronger project going forward we would want to finish mesh generation by converting the 3D line graph into closed polygonal faces. We would also want to add a proper evaluation by comparing our output against a ground truth model from a multi view photogrammetry tool like COLMAP. Supporting curved geometry like arches and domes would expand the range of buildings the system can handle. Finally, connecting the output to a real time 3D renderer would make the system much more useful and easier to demonstrate.
