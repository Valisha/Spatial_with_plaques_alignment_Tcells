# Xenium spatial transcriptomics with plaque alignment and immune-cell analysis

This repository contains a Scanpy/Squidpy workflow for integrating 10x Genomics Xenium spatial transcriptomics with post-Xenium immunofluorescence imaging. The analysis was developed for APP/PS1 infected and mock mouse-brain sections and includes quality control, cell annotation, image alignment, alpha-shape plaque detection, plaque-edge proximity labeling, and immune-cell analysis.

The newest plaque workflow is built around alpha shapes so plaque proximity is measured from the actual plaque boundary instead of assuming that every plaque is a perfect circle.

## Main workflow

| Step | Notebook | Purpose |
| --- | --- | --- |
| 1 | `01_preprocessing.ipynb` | Load Xenium outputs, calculate QC metrics, and filter infected, mock, and reference data. |
| 2 | `02_Calculating_initial_neighbours.ipynb` | Calculate initial spatial neighborhoods and export annotations for tissue review. |
| 3 | `03_Remove_bad_tissues.ipynb` | Remove manually identified poor-quality tissue regions. |
| 4 | `04_SCVI.ipynb` | Train scVI, construct the latent-space neighbor graph, run UMAP, and cluster cells. |
| 5 | `05_SCANVI_LabelTransfer_Annotations.ipynb` | Transfer reference labels with scANVI/scPoli. |
| 6 | `05_manual_annotations.ipynb` | Review marker genes and assign final manual cell-type annotations. |
| 7 | `06_plaques_alignment_final_07082026.ipynb` | Align the post-Xenium IF image with the infected and mock Xenium coordinate systems. |
| 8 | `08_plaque_detected_alpha_pipeline_clean.ipynb` | Detect plaques with an alpha-shape boundary workflow, calculate signed plaque-edge distances, and export proximity labels. |
| 9 | `09_cell_analysis_from_plaques.ipynb` | Analyze cell composition near plaques, classify T-cell-positive plaques, and compare plaque-associated immune cells. |

## Newest update: alpha-shape plaque-edge proximity

The alpha-shape workflow lives in:

- `08_plaque_detected_alpha_pipeline_clean.ipynb`: output-free GitHub copy of the latest workflow.
- `docs/alphashape_plaque_edge_workflow.md`: notes on how the method works and what the analysis is trying to achieve.

Alpha shapes are useful here because plaques are irregular tissue structures. A center-and-radius approximation can blur the biology by treating every plaque as round, even when the IF signal has concave edges, branches, or merged regions. The alpha-shape workflow instead:

1. thresholds the plaque-positive IF signal inside the aligned infected tissue;
2. cleans the binary plaque mask and optionally separates touching plaques with watershed;
3. builds a concave alpha-shape boundary around each plaque-positive region;
4. samples the plaque circumference as a boundary point cloud;
5. assigns each cell to the nearest plaque boundary using a nearest-neighbor query; and
6. writes signed edge distances and cumulative 15, 20, 30, and 40 um proximity labels.

The goal is to ask which immune cells are inside plaques, sitting directly along plaque edges, or enriched in near-plaque neighborhoods. This should make downstream T-cell and microglia comparisons more biologically honest because distances are measured from the plaque outline rather than a simplified plaque center.

## Supporting and exploratory notebooks

- `08_Intensity_based_plaque_detection_um_diameter.ipynb`: earlier intensity-based plaque detection and circular edge-distance workflow.
- `08_plaque_proximity_analysis.py`: script version of the earlier plaque-proximity assignment logic.
- `06_plaques_alignment.ipynb`: reusable alignment workflow and helper functions.
- `06_plaques_alignment_VS.ipynb`: alternate alignment development notebook.
- `07_identify_plaques.ipynb`: earlier plaque segmentation and cell-assignment workflow.
- `07_Intensity_based_plaque_detection.ipynb`: earlier intensity-based plaque detection notebook.
- `07_image_realignment_JR_SCRIPT.ipynb`: image realignment workflow based on an external alignment script.
- `DIY_spatial_20260722.ipynb`: end-to-end exploratory Scanpy/Squidpy workflow.

## Plaque detection and distance definition

Plaques are detected from post-Xenium IF channel 0 after image alignment into the Xenium coordinate system. The newest workflow uses the plaque-positive IF mask to generate alpha-shape plaque boundaries. Distance is then measured from the plaque circumference:

```text
signed plaque-edge distance < 0  = cell centroid is inside a plaque
signed plaque-edge distance = 0  = cell centroid is on the plaque boundary
signed plaque-edge distance > 0  = cell centroid is outside the plaque
```

Cells are assigned to their nearest plaque boundary and binned into cumulative distance labels:

- `within_15um_of_plaque`
- `within_20um_of_plaque`
- `within_30um_of_plaque`
- `within_40um_of_plaque`

The final annotated object also stores `plaque_edge_distance_um`, `nearest_plaque_id`, `inside_plaque`, `plaque_proximity_valid`, and `plaque_edge_cutoff_um` so the plaque proximity calls can be reused downstream.

## T-cell and microglia analysis

In the current manual annotation, T cells are represented by the `Lymphocytes` label. Plaque-edge labels can be used to ask whether lymphocytes, microglia states, or other immune populations accumulate inside plaques or within defined edge-distance bands.

The downstream notebooks compare cell-type composition and gene-expression patterns near plaques. These cell-level tests are exploratory because plaques and cells from one tissue section are not independent biological replicates.

## Main Python packages

- `scanpy`
- `anndata`
- `squidpy`
- `scvi-tools`
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`
- `scikit-image`
- `tifffile`
- `alphashape`
- `shapely`

The notebooks were run with Python 3.11. Package versions should be recorded in the analysis environment before reproducing the workflow.

## Data configuration

The notebooks contain paths specific to the original workstation and Box storage layout. Before running them, update the input and output path cells for:

- infected and mock Xenium output directories;
- the combined post-Xenium OME-TIFF;
- infected and mock alignment matrices;
- intermediate and final H5AD files; and
- results directories.

Raw Xenium outputs, OME-TIFF images, H5AD results, generated figures, and generated proximity CSVs are not intended to be stored in this repository.

## Recommended execution order

Run the canonical notebooks sequentially from steps 1 through 9. Restart the Jupyter kernel before running a notebook from the top so stale variables from exploratory cells do not affect coordinate transformations, plaque filtering, or edge-distance assignments.
