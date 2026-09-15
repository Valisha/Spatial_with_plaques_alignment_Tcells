# Why the two IF pixel-size results differ

The alignment matrix is the same in both notebooks. Its linear part has an IF-to-Xenium scale of about `3.0521 Xenium pixels / full-resolution IF pixel`. At `0.2125 µm / Xenium pixel`, a full-resolution IF pixel is therefore:

`3.0521 × 0.2125 = 0.6486 µm`.

The differing results come from the OME-TIFF pyramid level:

| Notebook | Code actually used | Image shape (Y, X) | Downsample | Result |
|---|---:|---:|---:|---:|
| `08_plaque_detected_alpha_pipeline.ipynb` | `read_ome_level(..., level=0)` | `(9285, 10203)` | `(1, 1)` | `0.649 µm/pixel`, `0.421 µm²/pixel` |
| `08_Intensity_based_plaque_detection_um_diameter.ipynb` | `read_ome_level(..., level=PLAQUE_LEVEL)` where `PLAQUE_LEVEL=1` | `(4642, 5101)` | about `(2.0002, 2.0002)` | `1.297 µm/pixel`, `1.683 µm²/pixel` |

Thus level 1 pixels are twice as wide and tall as level 0 pixels. Their area is four times as large. The alpha notebook's `PLAQUE_LEVEL = 1` is misleading because the next call hard-codes `level=0`.

## Other issues found

1. `pixel_size_um = 0.2125` in the alpha notebook is initially the **Xenium** pixel size, not the registered IF pixel size. Reusing a generic name makes it easy to mix coordinate systems.
2. The alpha notebook is currently not an alpha-shape pipeline: it ends immediately after the IF pixel-size calculation and contains neither `alphashape` nor Delaunay triangulation.
3. `infected_tissue_mask = image_stack[0] > 0` is not a meaningful tissue mask for a normal intensity image because almost every nonzero background pixel becomes tissue. The later cell-coordinate hull is more defensible, though a concave tissue boundary may be preferable.
4. The product of x/y vector lengths is only an exact pixel area when affine columns are perpendicular. The streamlined code uses `abs(det(A))`, which remains correct if the alignment contains shear.
5. A single scalar size should be treated as an equivalent size. The streamlined code uses `sqrt(pixel_area)` and retains separate x and y resolutions for anisotropic images.

## Streamlined workflow

The companion script `08_streamlined_if_scale_and_plaque_detection.py` separates the workflow into small steps:

1. **Read one OME pyramid level** and calculate its x/y downsample relative to level 0.
2. **Load the IF→Xenium matrix.** Its inverse maps Xenium pixels back into full-resolution IF pixels.
3. **Map cells:** Xenium µm → Xenium pixels → full-resolution IF pixels → selected-level IF pixels.
4. **Calculate scale:** transform one selected-level pixel step through the matrix and convert Xenium pixels to µm. Use the affine determinant for area.
5. **Prepare plaque signal:** subtract broad background from locally smoothed intensity inside a tissue mask.
6. **Choose a plaque model:**
   - `detect_round_plaques(...)` uses Laplacian-of-Gaussian circles.
   - `segment_plaque_candidates(...)` followed by `detect_nonround_plaques(...)` produces alpha-shape polygons.
7. **Choose the alpha implementation:** `backend="alphashape"` uses the `alphashape` package; `backend="delaunay"` explicitly filters SciPy Delaunay triangles by circumradius. Both keep a true polygon boundary rather than converting plaques to circles.

For this dataset, set `PLAQUE_LEVEL = 1` consistently if the lower-memory `(4642, 5101)` image is intended. Use `PLAQUE_LEVEL = 0` consistently only when full-resolution detection is intended.
