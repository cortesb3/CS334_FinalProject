# Vision-Guided Procedural Modeling

**Team Members:** Albert Wu, Bea Cortes, Eli Dubizh, Emma Tarrence

---

## Background and Motivation

Creating an accurate 3D model of a real-world building is harder than it looks. Professional methods typically require either a fleet of cameras set up around the structure, expensive LiDAR scanners that cost tens of thousands of dollars, or many hours of manual work in tools like Blender where someone traces and sculpts the geometry by hand. None of these options are accessible to a student, a small studio, or someone who just wants a quick 3D approximation of a building they photographed on their phone.

We wanted to see how far we could push a pipeline that starts with just a single photograph. The core question driving the project is: *can you give the system one image and get back something usable in 3D?* That's a genuinely hard problem because a flat photo throws away depth information — the camera collapses a three-dimensional scene onto a two-dimensional sensor, and recovering what was lost requires either multiple viewpoints or some kind of intelligent guess about depth.

This project interested us because it sits at the intersection of classical computer vision (edge detection, line finding) and modern AI (neural monocular depth estimation), and the output — a 3D structural skeleton of a building — has real uses in game development, architecture visualization, and simulation. Getting it to work with a single image keeps the barrier to entry low.

---

## Related Work / Prior Art

Several approaches exist for turning 2D images into 3D geometry, each with different trade-offs.

**Monocular Depth Estimation** is the technique of predicting how far away each pixel is from the camera using a single image. Intel's MiDaS model (Ranftl et al., 2020) is one of the most widely used tools for this. It was trained on a huge mix of datasets and generalizes well to real-world photos. The output is a relative depth map — it tells you which things are closer and which are farther, but not exact metric distances in meters. We use MiDaS as one of our building blocks.

**Direct 3D generation via generative AI** is an emerging area where models like Point-E or Shap-E (OpenAI) try to produce full 3D meshes from image input directly. These work surprisingly well on isolated objects like chairs or mugs, but they are basically a black box — you have little control over the output, and they tend to struggle with large architectural structures.

**Procedural and semantic reconstruction** approaches, like those used in CityEngine (Esri) or work from the ETH Zurich group on building reconstruction from floor plans and facades, try to identify meaningful semantic pieces (walls, windows, roof edges) and reconstruct geometry using rule-based or grammar-based systems. These are very accurate but typically require significant manual annotation or specialized training data.

Our approach is closest to that third category, but we avoid the need for any pre-labeled data by using classical computer vision for feature extraction combined with AI depth to fill in the Z-axis.

---

## Technical Approach

Our pipeline is split into three modules that pass data to each other through JSON files: the **vision module**, the **depth module**, and the **geometry module**.

### Module 1: Vision Pipeline (2D Feature Extraction)

The vision module (`src/vision/vision_extractor.py`) takes a building photograph as input and extracts its structural skeleton as 2D features — lines and corners — stored in a JSON file.

The processing steps are:

1. **Preprocessing:** The image is converted to grayscale and then blurred with a 13×13 Gaussian kernel (sigma=2.0). This aggressive blur is intentional — it suppresses fine texture like brick siding, tree leaves, or glass reflections that would otherwise confuse the edge detector into finding thousands of spurious edges.

2. **Adaptive Canny Edge Detection:** Rather than using fixed thresholds for the Canny edge detector, we compute them dynamically from the image's median pixel intensity using a sigma factor of 0.33. This makes the algorithm adapt to different lighting conditions and building materials without needing manual parameter tuning per image.

3. **Probabilistic Hough Line Transform:** We run OpenCV's `HoughLinesP` with strict parameters: a vote threshold of 100, minimum line length of 90 pixels, and maximum gap of 40 pixels. This filters out short spurious edges and only accepts lines that are long, continuous, and well-supported by edge votes. In testing on the Purdue Bell Tower image, this reduced the raw line count from roughly 400 down to about 45.

4. **Corner Detection:** We use `goodFeaturesToTrack` with a quality level of 0.02 and a minimum inter-corner distance of 25 pixels to find structural vertices in the image. A post-processing step removes any duplicate corners within 30 pixels of each other.

**Output:** A JSON file containing the image dimensions, a list of detected line segments (start and end pixel coordinates), and a list of detected corners (pixel x, y coordinates).

### Module 2: Depth Estimation (Adding the Z-Axis)

The depth module (`src/depth/depth_estimator.py` and `enrich_manifest.py`) takes the vision JSON and the original image, runs a neural depth estimator, and annotates every feature with a Z-coordinate.

We use Intel's **MiDaS Small** model loaded via PyTorch Hub. This variant is fast and produces good relative depth for architectural scenes. The model outputs a depth map in arbitrary units, which we normalize to the [0, 1] range and then invert — so a value of 1.0 means "closest to the camera" and 0.0 means "farthest away." This inversion makes the values more intuitive for downstream use.

For each corner `(x, y)` in the vision JSON, we sample the depth map at that pixel location to get a Z value. For each line, we sample at both endpoints. The result is an enriched JSON where every feature now has three coordinates.

**Output:** An enriched JSON with corners as `{x, y, z}` and line endpoints as `{x1, y1, z1, x2, y2, z2}`.

### Module 3: Geometry (Coordinate Normalization)

The geometry module (`src/geometry/manifest_parser.py`) converts the pixel-space coordinates from the enriched JSON into world-space 3D coordinates that can be handed off to a mesh generation system.

The conversion works as follows:

```
world_x =  (pixel_x / image_width  - 0.5) * world_scale
world_y = -(pixel_y / image_height - 0.5) * world_scale
world_z =   depth_z * depth_scale
```

The Y axis is flipped because in image space Y increases downward, while in standard 3D world space Y increases upward. The default `world_scale` is 10.0, mapping the image to a [-5, +5] range on both X and Y. The default `depth_scale` is 5.0, mapping the [0, 1] depth range to [0, 5] world units.

The parser handles both the basic (no depth) and depth-enriched JSON formats, defaulting Z to 0 when no depth data is present.

**Output:** A Python dictionary with `corners_3d` and `lines_3d` lists containing tuples of (x, y, z) world coordinates, ready to be consumed by the mesh generation step.

### Mesh Generation and Rendering (In Progress)

The mesh generation step — connecting the 3D line segments and corners into actual polygonal faces, building topology, and exporting to a 3D format like OBJ or GLTF — has not been fully implemented yet. The coordinate normalization from the parser is complete and correct; the remaining work is building the graph structure from the line endpoints, identifying building faces from line intersections, and running a surface reconstruction algorithm to produce a valid mesh. No rendering pipeline exists yet beyond the 2D visualizations.

---

## Timeline and Development Process

The project was split into parallel tracks to let team members work independently on different modules.

**Early phase:** Albert and Bea built the vision pipeline. Initial attempts used fixed Canny thresholds and returned hundreds of noisy lines. Several rounds of parameter tuning — increasing the Hough vote threshold, raising the minimum line length, adding the Gaussian blur — progressively reduced noise. The Purdue Bell Tower was the primary test subject throughout this phase because it has a mix of easy geometry (straight tower walls) and hard geometry (clock faces, decorative arches).

**Mid phase:** Emma integrated the MiDaS depth model to add Z-coordinates to the JSON output. This included writing the enrichment script that samples the depth map at feature coordinates and producing depth visualizations (false-color depth maps using the Inferno colormap) to verify correctness.

**Recent phase:** Eli wrote the manifest parser to normalize coordinates to world space, and the team integrated all three modules into a unified pipeline script (`src/depth/pipeline.py`). The main current milestone still pending is mesh generation, which will connect the normalized 3D points into renderable geometry.

---

## Current Results

The vision pipeline successfully extracts structural line and corner features from building photographs. On the Purdue Bell Tower test image, it reduces around 400 raw detected edges down to roughly 45 meaningful structural lines and 60 corners. A visualization overlays the detections on the original image — lines in green, corners as red circles — making it easy to verify that the extracted skeleton corresponds to real building edges rather than texture noise.

The depth estimation module produces plausible relative depth maps for architectural scenes. Close foreground elements (lower tower base, nearby brickwork) correctly appear as higher depth values, while distant background sky regions appear as lower values.

The manifest parser correctly converts pixel coordinates to world-space 3D coordinates, verified by checking that bounding boxes of the resulting point cloud match expected ranges.

The system does not currently produce renderable 3D geometry — the pipeline ends at a structured list of 3D line segments and corner points in world space. Full mesh output is the next milestone.

---

## Video and Code Links

**Codebase:** [https://github.com/CS334FinalProject/vision-guided-procedural-modeling](https://github.com/CS334FinalProject/vision-guided-procedural-modeling) *(replace with actual link)*

**Demo Video:** *(link to be added before submission)*

---

## Limitations and Future Work

**What works:** The vision pipeline reliably extracts straight structural edges from building photographs. The depth model adds plausible Z-values to each feature. The coordinate normalization produces valid world-space 3D points.

**What does not work yet:** Mesh generation is unfinished, so there is no actual 3D output yet. The vision pipeline struggles with circular architectural features (clock faces, decorative roundels) because the Hough Line Transform is fundamentally designed for straight lines. Repetitive textures like brick can still slip through the noise filter and produce spurious detections. The MiDaS model produces only *relative* depth — it can tell you that the top of the tower is farther away than the base, but it cannot tell you the actual distance in meters, so the scale of the resulting 3D model is not physically accurate.

**What would make this a stronger research contribution:**

- *Mesh generation:* Completing this step is the most important remaining task. The connected line graph needs to be converted into closed polygonal faces using something like alpha shapes or a building-specific topology inference algorithm.

- *Better evaluation:* Right now there is no quantitative evaluation — we do not compare our output to any ground-truth 3D model. Adding a comparison to a photogrammetry result (e.g., using RealityCapture or COLMAP on multi-view photos of the same building) would give a real sense of how accurate the output is.

- *Absolute depth scale:* MiDaS gives relative depth. Incorporating a known reference object in the scene (like a person of average height) or using GPS metadata from the phone photo could help calibrate the depth to real-world units.

- *Handling curved geometry:* The current pipeline only models straight-line architecture. Extending it to handle arches, domes, and circular features would dramatically expand the range of buildings it can reconstruct.

- *End-to-end rendering:* Connecting the output to a real-time renderer (e.g., Three.js for web or a simple OpenGL viewer) would let users interactively explore the reconstructed model, making the system far more useful and demonstrable.

- *User study or qualitative comparison:* Having people judge whether the output "looks like" the input building, compared against a manual Blender model of the same structure, would provide a more subjective but useful measure of success.
