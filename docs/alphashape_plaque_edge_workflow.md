# Alpha-shape plaque-edge proximity workflow

Last updated: September 14, 2026

## What I am trying to achieve

I am trying to make plaque-neighborhood analysis more faithful to the real tissue image. Plaques do not always look like clean circles, so measuring distance from plaque centers or equivalent radii can hide important spatial structure. This workflow measures from the alpha-shape plaque edge instead.

The goal is to label cells by where they sit relative to each plaque:

- inside a plaque;
- directly at the plaque boundary;
- within 15, 20, 30, or 40 um of the plaque edge; or
- farther away from the plaque neighborhood.

Those labels can then be used to ask whether T cells, microglia states, or other immune populations accumulate around disease-relevant plaque regions.

## How alpha shape works

An alpha shape is a flexible boundary around a set of points. It is related to a convex hull, but it can bend inward and preserve concave structure. That matters here because plaque-positive IF signal can have irregular edges, touching regions, and uneven shapes.

In this workflow, alpha shape is not the first plaque detector. The pipeline first identifies plaque-positive pixels from the IF image, cleans the binary mask, and optionally separates touching plaques with watershed. Alpha shape is then used to trace a tighter plaque boundary around each candidate plaque region.

The `alpha` setting controls how tightly the boundary follows the points. In the notebook, the radius-based parameterization is documented as:

```text
smaller alpha value -> closer to a convex hull
larger alpha value  -> tighter, more concave boundary
```

The current workflow records the alpha-shape parameters in the output object so the plaque calls can be reviewed and reproduced.

## Workflow summary

1. Load Xenium spatial coordinates and the post-Xenium IF image.
2. Align cell coordinates into the IF image coordinate space.
3. Select the plaque-positive IF channel.
4. Normalize and threshold the plaque signal.
5. Clean the mask by removing small objects and filling holes.
6. Optionally split touching plaques using watershed.
7. Build an alpha-shape boundary for each plaque-positive region.
8. Convert each plaque boundary into a dense circumference point cloud.
9. Use `scipy.spatial.cKDTree` to find the nearest plaque boundary for every cell.
10. Store signed plaque-edge distances and cumulative proximity labels.

## Distance interpretation

The distance value is signed:

```text
negative distance = cell centroid is inside a plaque
zero distance     = cell centroid is on the plaque edge
positive distance = cell centroid is outside the plaque
```

This lets the analysis distinguish plaque interior from plaque-adjacent neighborhoods.

## Output annotations

The workflow stores the following key fields:

- `plaque_edge_distance_um`
- `nearest_plaque_id`
- `plaque_proximity_valid`
- `inside_plaque`
- `plaque_edge_cutoff_um`
- `within_15um_of_plaque`
- `within_20um_of_plaque`
- `within_30um_of_plaque`
- `within_40um_of_plaque`

The output also stores plaque-level tables and plaque-boundary coordinates so the distance labels can be inspected later without reloading the original IF image.

## Biological question

The broader question is whether disease-associated immune states organize around plaques in spatially meaningful ways. With alpha-shape plaque-edge distances, I can ask whether specific T-cell or microglia phenotypes are enriched right at plaque boundaries compared with cells farther away.
