from pathlib import Path
import re

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
INFECTED_H5AD = SCRIPT_DIR / "cluster_annotation/Infected/h5ad/infected_cluster_annotation_scRNA_style.h5ad"
MOCK_H5AD = SCRIPT_DIR / "cluster_annotation/Mock/h5ad/infected_cluster_annotation_scRNA_style.h5ad"
GENE_GUIDE_PATH = Path(
    "~/Library/CloudStorage/Box-Box/Kaech Lab Folder/Valisha/"
    "Gene List of sub-clustering/BRAIN_CLUSTERING_GUIDE_IRENE_Vs.xlsx"
).expanduser()
OUTPUT_DIR = SCRIPT_DIR / "cluster_annotation/together"


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def cluster_sort_key(value):
    try:
        return 0, int(value)
    except ValueError:
        return 1, str(value)


def make_output_folders(output_dir):
    folders = {
        "root": Path(output_dir),
        "deg": Path(output_dir) / "DEG_results",
        "heatmap": Path(output_dir) / "heatmaps",
        "dotplot": Path(output_dir) / "dotplots",
        "feature": Path(output_dir) / "feature_plots",
        "violin": Path(output_dir) / "violin_plots",
        "umap": Path(output_dir) / "umap_plots",
        "qc": Path(output_dir) / "QC_plots",
        "h5ad": Path(output_dir) / "h5ad",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def load_and_merge(infected_path, mock_path):
    infected = sc.read_h5ad(infected_path)
    mock = sc.read_h5ad(mock_path)

    if not infected.var_names.equals(mock.var_names):
        shared_genes = infected.var_names.intersection(mock.var_names)
        if len(shared_genes) == 0:
            raise ValueError("The infected and mock objects have no shared genes.")

        print(f"Using {len(shared_genes)} shared genes.")
        infected = infected[:, shared_genes].copy()
        mock = mock[:, shared_genes].copy()

    if "counts" not in infected.layers or "counts" not in mock.layers:
        raise ValueError("Both input objects must contain raw counts in layers['counts'].")

    infected.obs["condition"] = "infected"
    mock.obs["condition"] = "mock"

    combined = ad.concat(
        {"infected": infected, "mock": mock},
        axis=0,
        join="inner",
        label="condition_source",
        index_unique="-",
        merge="same",
    )

    combined.obs["condition"] = combined.obs["condition"].astype("category")
    combined.obs["condition_source"] = combined.obs["condition_source"].astype("category")
    combined.layers["counts"] = combined.layers["counts"].copy()
    combined.X = combined.layers["counts"].copy()

    for key in list(combined.obsm.keys()):
        if key.startswith("X_"):
            del combined.obsm[key]

    for key in list(combined.varm.keys()):
        if key == "PCs":
            del combined.varm[key]

    combined.obsp.clear()
    combined.uns.clear()

    print(combined)
    print(combined.obs["condition"].value_counts())
    return combined


def preprocess_combined(
    adata,
    cluster_key="leiden_combined",
    target_sum=1e4,
    n_pcs=50,
    n_neighbors=15,
    resolution=0.5,
    random_state=0,
):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    usable_pcs = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=usable_pcs, svd_solver="arpack", random_state=random_state)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=usable_pcs, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)

    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=cluster_key,
            random_state=random_state,
            flavor="igraph",
            n_iterations=2,
        )
    except (ImportError, TypeError):
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=cluster_key,
            random_state=random_state,
        )

    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    return adata


def save_pca_umap_and_composition(adata, folders, cluster_key="leiden_combined"):
    sc.pl.pca_variance_ratio(adata, n_pcs=min(50, adata.obsm["X_pca"].shape[1]), log=True, show=False)
    plt.gcf().savefig(folders["qc"] / "combined_PCA_variance_ratio.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    axes = sc.pl.umap(
        adata,
        color=[cluster_key, "condition"],
        frameon=False,
        size=2,
        wspace=0.4,
        show=False,
    )
    figure = axes[0].figure if isinstance(axes, list) else axes.figure
    figure.savefig(folders["umap"] / "combined_UMAP_clusters_and_condition.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    cluster_counts = pd.crosstab(adata.obs[cluster_key], adata.obs["condition"])
    cluster_percentages = pd.crosstab(
        adata.obs[cluster_key],
        adata.obs["condition"],
        normalize="columns",
    ) * 100

    cluster_counts.to_csv(folders["root"] / "cluster_counts_by_condition.csv")
    cluster_percentages.to_csv(folders["root"] / "cluster_percentages_by_condition.csv")

    ax = cluster_percentages.plot(kind="bar", figsize=(12, 6), width=0.85)
    ax.set_xlabel("Combined Leiden cluster")
    ax.set_ylabel("Percent of condition")
    ax.set_title("Cluster composition by condition")
    ax.legend(title="Condition", frameon=False)
    plt.tight_layout()
    plt.savefig(folders["qc"] / "cluster_composition_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close()


def compute_and_save_cluster_markers(adata, folders, cluster_key="leiden_combined"):
    rank_key = "combined_cluster_markers"
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method="wilcoxon",
        pts=True,
        use_raw=False,
        key_added=rank_key,
    )

    groups = list(adata.uns[rank_key]["names"].dtype.names)
    tables = []

    for group in groups:
        table = sc.get.rank_genes_groups_df(adata, group=group, key=rank_key)
        table.insert(0, "cluster", str(group))
        tables.append(table)

    all_markers = pd.concat(tables, ignore_index=True)
    significant_markers = all_markers[
        (all_markers["pvals_adj"] < 0.05)
        & (all_markers["logfoldchanges"] > 0.25)
    ].copy()
    significant_markers = significant_markers.sort_values(
        ["cluster", "logfoldchanges"], ascending=[True, False]
    )
    top_markers = significant_markers.groupby("cluster", observed=True).head(20)

    excel_path = folders["deg"] / f"{cluster_key}_marker_genes.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        all_markers.to_excel(writer, sheet_name="all_markers", index=False)
        significant_markers.to_excel(writer, sheet_name="significant_markers", index=False)
        top_markers.to_excel(writer, sheet_name="top_20_markers", index=False)

        for group in groups:
            all_markers.loc[all_markers["cluster"] == str(group)].to_excel(
                writer,
                sheet_name=f"cluster_{safe_name(group)}"[:31],
                index=False,
            )

    all_markers.to_csv(folders["deg"] / f"{cluster_key}_all_marker_genes.csv", index=False)
    significant_markers.to_csv(
        folders["deg"] / f"{cluster_key}_significant_marker_genes.csv", index=False
    )
    return all_markers, significant_markers, top_markers


def load_gene_guide(gene_guide_path):
    guide = pd.read_excel(gene_guide_path, sheet_name="Long_format_all_genes")
    guide = guide.dropna(subset=["Type", "Gene"]).copy()
    guide["Type"] = guide["Type"].astype(str)
    guide["Gene"] = guide["Gene"].astype(str)
    return guide


def save_gene_guide_visualizations(
    adata,
    gene_guide,
    folders,
    cluster_key="leiden_combined",
    genes_per_feature_figure=4,
    genes_per_dotplot=20,
    genes_per_violin=6,
):
    cluster_order = sorted(
        adata.obs[cluster_key].cat.categories.astype(str), key=cluster_sort_key
    )

    if f"{cluster_key}_colors" not in adata.uns:
        sc.pl.umap(adata, color=cluster_key, show=False)
        plt.close("all")

    categories = adata.obs[cluster_key].cat.categories.astype(str)
    cluster_colors = dict(zip(categories, adata.uns[f"{cluster_key}_colors"]))

    for celltype in gene_guide["Type"].unique():
        genes = gene_guide.loc[gene_guide["Type"] == celltype, "Gene"].drop_duplicates()
        genes = [gene for gene in genes if gene in adata.var_names]

        if not genes:
            print(f"No guide genes found for {celltype}")
            continue

        for chunk_number, start in enumerate(range(0, len(genes), genes_per_feature_figure), 1):
            gene_set = genes[start : start + genes_per_feature_figure]
            sc.pl.umap(
                adata,
                color=gene_set,
                ncols=2,
                size=2,
                frameon=False,
                show=False,
            )
            figure = plt.gcf()
            figure.suptitle(f"{celltype} markers", fontsize=14, y=1.01)
            figure.savefig(
                folders["feature"] / f"{safe_name(celltype)}_feature_set_{chunk_number}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(figure)

        for chunk_number, start in enumerate(range(0, len(genes), genes_per_dotplot), 1):
            gene_set = genes[start : start + genes_per_dotplot]
            dotplot = sc.pl.dotplot(
                adata,
                var_names=gene_set,
                groupby=cluster_key,
                categories_order=cluster_order,
                dendrogram=False,
                cmap="coolwarm",
                standard_scale="var",
                figsize=(max(8, len(gene_set) * 0.45), max(5, len(cluster_order) * 0.35)),
                return_fig=True,
                show=False,
            )
            dotplot.savefig(
                folders["dotplot"] / f"{safe_name(celltype)}_dotplot_set_{chunk_number}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close("all")

        for chunk_number, start in enumerate(range(0, len(genes), genes_per_violin), 1):
            gene_set = genes[start : start + genes_per_violin]
            sc.pl.violin(
                adata,
                keys=gene_set,
                groupby=cluster_key,
                order=cluster_order,
                palette=cluster_colors,
                rotation=90,
                stripplot=False,
                multi_panel=True,
                show=False,
            )
            figure = plt.gcf()
            figure.suptitle(f"{celltype} marker expression", fontsize=14, y=1.02)
            figure.savefig(
                folders["violin"] / f"{safe_name(celltype)}_violin_set_{chunk_number}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(figure)


def save_average_marker_heatmap(
    adata,
    all_markers,
    folders,
    cluster_key="leiden_combined",
    top_genes_per_cluster=10,
):
    heatmap_genes = (
        all_markers.sort_values(["cluster", "logfoldchanges"], ascending=[True, False])
        .groupby("cluster", observed=True)
        .head(top_genes_per_cluster)["names"]
        .drop_duplicates()
        .tolist()
    )
    heatmap_genes = [gene for gene in heatmap_genes if gene in adata.var_names]

    expression = adata[:, heatmap_genes].to_df()
    expression["cluster"] = adata.obs[cluster_key].astype(str).to_numpy()
    average_expression = expression.groupby("cluster", observed=True).mean().T
    cluster_order = sorted(average_expression.columns.astype(str), key=cluster_sort_key)
    average_expression = average_expression.loc[:, cluster_order]
    average_z = average_expression.sub(average_expression.mean(axis=1), axis=0)
    average_z = average_z.div(
        average_expression.std(axis=1).replace(0, np.nan), axis=0
    ).fillna(0)

    categories = adata.obs[cluster_key].cat.categories.astype(str)
    colors = dict(zip(categories, adata.uns[f"{cluster_key}_colors"]))
    column_colors = pd.Series([colors[cluster] for cluster in cluster_order], index=cluster_order)

    heatmap = sns.clustermap(
        average_z,
        row_cluster=False,
        col_cluster=False,
        col_colors=column_colors,
        cmap="RdBu_r",
        center=0,
        figsize=(12, 16),
        xticklabels=True,
        yticklabels=True,
    )
    heatmap.ax_heatmap.set_xlabel("Combined Leiden cluster")
    heatmap.ax_heatmap.set_ylabel("Marker genes")
    heatmap.savefig(
        folders["heatmap"] / f"{cluster_key}_cluster_average_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(heatmap.fig)


def save_combined_h5ad(adata, folders):
    output_path = folders["h5ad"] / "infected_mock_combined_cluster_annotation.h5ad"
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.write_h5ad(output_path, compression="gzip")
    print(f"Saved combined H5AD: {output_path}")
    return output_path


def run_combined_analysis(
    infected_path=INFECTED_H5AD,
    mock_path=MOCK_H5AD,
    gene_guide_path=GENE_GUIDE_PATH,
    output_dir=OUTPUT_DIR,
    cluster_key="leiden_combined",
    resolution=0.5,
):
    folders = make_output_folders(output_dir)
    combined = load_and_merge(infected_path, mock_path)
    combined = preprocess_combined(
        combined,
        cluster_key=cluster_key,
        resolution=resolution,
    )
    save_pca_umap_and_composition(combined, folders, cluster_key=cluster_key)
    all_markers, significant_markers, top_markers = compute_and_save_cluster_markers(
        combined,
        folders,
        cluster_key=cluster_key,
    )
    gene_guide = load_gene_guide(gene_guide_path)
    save_gene_guide_visualizations(
        combined,
        gene_guide,
        folders,
        cluster_key=cluster_key,
    )
    save_average_marker_heatmap(
        combined,
        all_markers,
        folders,
        cluster_key=cluster_key,
    )
    h5ad_path = save_combined_h5ad(combined, folders)
    return combined, all_markers, significant_markers, top_markers, h5ad_path


if __name__ == "__main__":
    run_combined_analysis()
