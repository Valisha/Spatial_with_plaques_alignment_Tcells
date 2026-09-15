from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "plaque_results"
DISTANCES_UM = (5, 10, 15, 20, 25)


def annotate_cells(cells, plaques, distance_reference="edge"):
    required_cell_columns = {"cell_id", "x_um", "y_um"}
    required_plaque_columns = {
        "plaque_id",
        "x_xenium_um",
        "y_xenium_um",
        "radius_um",
    }

    missing_cells = required_cell_columns.difference(cells.columns)
    missing_plaques = required_plaque_columns.difference(plaques.columns)

    if missing_cells:
        raise ValueError(f"Cell table is missing columns: {sorted(missing_cells)}")
    if missing_plaques:
        raise ValueError(f"Plaque table is missing columns: {sorted(missing_plaques)}")
    if len(plaques) == 0:
        raise ValueError("The plaque table is empty.")

    cell_xy = cells[["x_um", "y_um"]].to_numpy(dtype=float)
    plaque_xy = plaques[["x_xenium_um", "y_xenium_um"]].to_numpy(dtype=float)

    tree = cKDTree(plaque_xy)
    center_distance_um, nearest_row = tree.query(cell_xy, k=1)
    nearest_plaques = plaques.iloc[nearest_row].reset_index(drop=True)

    annotated = cells.copy().reset_index(drop=True)
    annotated["nearest_plaque_id"] = nearest_plaques["plaque_id"].to_numpy()
    annotated["nearest_plaque_radius_um"] = nearest_plaques["radius_um"].to_numpy()
    annotated["distance_to_plaque_center_um"] = center_distance_um
    annotated["distance_to_plaque_edge_um"] = np.maximum(
        center_distance_um - annotated["nearest_plaque_radius_um"].to_numpy(),
        0,
    )

    distance_column = (
        "distance_to_plaque_edge_um"
        if distance_reference == "edge"
        else "distance_to_plaque_center_um"
    )
    annotated["plaque_distance_um"] = annotated[distance_column]

    for distance_um in DISTANCES_UM:
        annotated[f"within_{distance_um}_um"] = (
            annotated["plaque_distance_um"] <= distance_um
        )

    bin_edges = [-np.inf, *DISTANCES_UM, np.inf]
    bin_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", ">25"]
    annotated["plaque_distance_bin_um"] = pd.cut(
        annotated["plaque_distance_um"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    )

    return annotated, distance_column


def make_summary(annotated):
    total_cells = len(annotated)
    rows = []

    for distance_um in DISTANCES_UM:
        count = int(annotated[f"within_{distance_um}_um"].sum())
        rows.append(
            {
                "summary_type": "cumulative",
                "distance_group_um": f"within_{distance_um}",
                "cell_count": count,
                "percent_of_cells": 100 * count / total_cells,
            }
        )

    exclusive_counts = annotated["plaque_distance_bin_um"].value_counts(sort=False)
    for distance_group, count in exclusive_counts.items():
        rows.append(
            {
                "summary_type": "exclusive_bin",
                "distance_group_um": str(distance_group),
                "cell_count": int(count),
                "percent_of_cells": 100 * int(count) / total_cells,
            }
        )

    return pd.DataFrame(rows)


def plot_distance_bins(annotated, plaques, output_path):
    colors = {
        "0-5": "#d73027",
        "5-10": "#fc8d59",
        "10-15": "#fee08b",
        "15-20": "#91cf60",
        "20-25": "#1a9850",
        ">25": "#d9d9d9",
    }

    fig, ax = plt.subplots(figsize=(10, 9))

    for distance_group, color in colors.items():
        selected = annotated["plaque_distance_bin_um"].astype(str) == distance_group
        ax.scatter(
            annotated.loc[selected, "x_um"],
            annotated.loc[selected, "y_um"],
            s=0.4,
            c=color,
            alpha=0.65,
            label=f"{distance_group} µm",
            rasterized=True,
        )

    ax.scatter(
        plaques["x_xenium_um"],
        plaques["y_xenium_um"],
        s=16,
        facecolors="none",
        edgecolors="black",
        linewidths=0.7,
        label="Plaque centers",
    )
    ax.set_title("Infected cells by distance from the nearest plaque edge")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(markerscale=4, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(
        description="Assign infected Xenium cells to plaque-distance groups."
    )
    parser.add_argument(
        "--cells",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "infected_cells_for_plaque_proximity.csv",
    )
    parser.add_argument(
        "--plaques",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "balanced_plaque_candidates.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--distance-reference",
        choices=("edge", "center"),
        default="edge",
        help="Use plaque-edge distance (recommended) or plaque-center distance.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cells = pd.read_csv(args.cells)
    plaques = pd.read_csv(args.plaques)
    annotated, distance_column = annotate_cells(
        cells,
        plaques,
        distance_reference=args.distance_reference,
    )
    summary = make_summary(annotated)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = args.output_dir / "infected_cells_with_plaque_distances.csv"
    summary_path = args.output_dir / "plaque_distance_summary.csv"
    figure_path = args.output_dir / "plaque_distance_spatial_plot.png"

    annotated.to_csv(annotated_path, index=False)
    summary.to_csv(summary_path, index=False)

    for distance_um in DISTANCES_UM:
        selected = annotated[annotated[f"within_{distance_um}_um"]]
        selected.to_csv(
            args.output_dir / f"infected_cells_within_{distance_um}_um.csv",
            index=False,
        )

    plot_distance_bins(annotated, plaques, figure_path)

    print(f"Distance measured from: {distance_column}")
    print(summary.to_string(index=False))
    print(f"\nSaved annotated cells: {annotated_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved spatial plot: {figure_path}")


if __name__ == "__main__":
    main()
