# Xenium spatial transcriptomics with plaque alignment and immune-cell analysis

This repository contains a Scanpy/Squidpy workflow for integrating 10x Genomics Xenium spatial transcriptomics with post-Xenium immunofluorescence imaging. The analysis was developed for APP/PS1 infected and mock mouse-brain sections and includes quality control, cell annotation, image alignment, plaque detection, plaque-edge proximity labeling, and immune-cell analysis.

## Main workflow

| Step | Notebook | Purpose |
| --- | --- | --- |
| 1 | `01_preprocessing.ipynb` | Load Xenium outputs, calculate QC metrics, and filter infected, mock, and reference data. |
| 2 | `02_Calculating_initial_neighbours.ipynb` | Calculate initial spatial neighborhoods and export annotations for tissue review. |
| 3 | `03_Remove_bad_tissues.ipynb` | Remove manually identified poor-quality tissue regions. |
| 4 | `04_SCVI.ipynb` | Train scVI, construct the latent-space neighbor graph, run UMAP, and cluster cells. |
| 5 | `05_SCANVI_LabelTransfer_Annotations.ipynb` | Transfer reference labels with scANVI/scPoli. |
| 5 | `05_manual_annotations.ipynb` | Review marker genes and assign final manual cell-type annotations. |
| 6 | `06_plaques_alignment_final_07082026.ipynb` | Align the post-Xenium IF image with the infected and mock Xenium coordinate systems. |
| 7 | `07_Intensity_based_plaque_detection.ipynb` | Develop and inspect intensity-based plaque detection on IF channel 0. |
| 8 | `08_Intensity_based_plaque_detection_um_diameter.ipynb` | Final plaque detection, size filtering, plaque-edge distance calculation, and annotated H5AD export. |
| 9 | `09_cell_analysis_from_plaques.ipynb` | Analyze cell composition near plaques, classify T-cell-positive plaques, and compare plaque-associated immune cells. |

## Supporting and exploratory notebooks

- `06_plaques_alignment.ipynb`: reusable alignment workflow and helper functions.
- `06_plaques_alignment_VS.ipynb`: alternate alignment development notebook.
- `07_identify_plaques.ipynb`: earlier plaque segmentation and cell-assignment workflow.
- `07_image_realignment_JR_SCRIPT.ipynb`: image realignment workflow based on an external alignment script.
- `DIY_spatial_20260722.ipynb`: end-to-end exploratory Scanpy/Squidpy workflow.

## Plaque detection and distance definition

Plaques are detected from post-Xenium IF channel 0 using background removal and multiscale Laplacian-of-Gaussian blob detection. The final workflow filters small detections and retains the reviewed plaque set.

Cell proximity is measured from the detected plaque boundary rather than the plaque center:

```text
distance to plaque edge = max(distance to plaque center - plaque radius, 0)
```

Cells inside a plaque therefore have an edge distance of `0 µm`. The final annotated object contains cumulative labels for cells within `<15`, `<20`, `<25`, `<30`, and `<50 µm` of the nearest plaque edge, along with the nearest plaque ID and continuous edge distance.

## T-cell and microglia analysis

In the current manual annotation, T cells are represented by the `Lymphocytes` label. A plaque is classified as:

- **T-cell positive:** at least one lymphocyte is assigned within 50 µm of its edge.
- **T-cell negative:** no lymphocytes are assigned within 50 µm of its edge.

The downstream notebook compares microglia near T-cell-positive and T-cell-negative plaques and exports differential-expression tables and figures. These cell-level tests are exploratory because plaques and cells from one tissue section are not independent biological replicates.

## Main Python packages

- `scanpy`
- `anndata`
- `squidpy`
- `scvi-tools`
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`
- `scikit-image`
- `tifffile`

The notebooks were run with Python 3.11. Package versions should be recorded in the analysis environment before reproducing the workflow.

## Data configuration

The notebooks contain paths specific to the original workstation and Box storage layout. Before running them, update the input and output path cells for:

- infected and mock Xenium output directories;
- the combined post-Xenium OME-TIFF;
- infected and mock alignment matrices;
- intermediate and final H5AD files; and
- results directories.

Raw Xenium outputs, OME-TIFF images, H5AD results, and generated figures are not intended to be stored in this repository.

## Recommended execution order

Run the canonical notebooks sequentially from steps 1 through 9. Restart the Jupyter kernel before running a notebook from the top so stale variables from exploratory cells do not affect coordinate transformations or plaque filtering.
