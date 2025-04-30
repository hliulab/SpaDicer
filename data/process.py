import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from scipy import stats

def preporcess(
        sc_adata,
        st_adata,
        st_type: str = 'spot',
        n_features: int = 2000,
        normalize: bool = True,
        select_hvg: str = 'intersection'
):
    """
    pre-process the scRNA-seq and spatial transcriptomics data (find HVGs and normalized the data)
    :param select_hvg: 'intersection' or 'union'
    :param sc_adata: AnnData object of scRNA-seq data
    :param st_adata: AnnData object of spatial transcriptomics data
    :param st_type: the type of spatial transcriptomics data, `spot` or `image`
    :param n_features: the number of HVGs to select
    :param normalize: whether to normalize the data or not
    :return: AnnData object of processed scRNA-seq data and spatial transcriptomic data
    """

    assert sc_adata.shape[1] >= n_features, 'There are too few genes in scRNA-seq data, please check again!'
    sc.pp.highly_variable_genes(sc_adata, flavor="seurat_v3", n_top_genes=n_features)

    assert st_type in ['spot',
                       'image'], 'Please select the correct type of spatial transcriptomic data, `spot` or `image`!'

    if st_type == 'spot':
        assert st_adata.shape[1] >= n_features, 'There are too few genes in ST data, please check again!'
        sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=n_features)
    elif st_type == 'image':
        if st_adata.shape[1] >= n_features:
            sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=n_features)
        else:
            sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=st_adata.shape[1])

    if normalize:
        # sc_adata
        sc.pp.normalize_total(sc_adata, target_sum=1e4)
        sc.pp.log1p(sc_adata)
        # st_adata
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata)

    sc_adata.raw = sc_adata
    st_adata.raw = st_adata

    sc_hvg = sc_adata.var['highly_variable'][sc_adata.var['highly_variable'] == True].index
    st_hvg = st_adata.var['highly_variable'][st_adata.var['highly_variable'] == True].index
    if select_hvg == 'intersection':
        inter_gene = set(sc_hvg).intersection(set(st_hvg))
    elif select_hvg == 'union':
        sc_gene = set(sc_adata.var_names)
        st_gene = set(st_adata.var_names)
        common_gene = set(sc_gene).intersection(set(st_gene))

        inter_gene = set(sc_hvg).union(set(st_hvg))
        inter_gene = set(inter_gene).intersection(set(common_gene))

    sc_adata = sc_adata[:, list(inter_gene)]
    st_adata = st_adata[:, list(inter_gene)]

    print('Data have been pre-processed.')
    return sc_adata, st_adata



def combined_spatial_clustering(expression_paths, spatial_paths, output_paths, n_clusters=5):
    """
    Load gene expression data and spatial location information for three slices, merge them,
    and perform clustering, ensuring consistent cluster labels.

    Parameters:
    - expression_paths: List of paths for the gene expression data for three slices.
    - spatial_paths: List of paths for the spatial location information for three slices.
    - output_paths: List of output paths for the clustering results of the three slices.
    - n_clusters: The number of clusters for the KMeans algorithm, default is 5.
    """

    # 1. Load the data for the first slice
    expression_data_1 = pd.read_csv(expression_paths[0], sep='\t', index_col=0)
    spatial_data_1 = pd.read_csv(spatial_paths[0], index_col=0)

    # 2. Load the data for the second slice
    expression_data_2 = pd.read_csv(expression_paths[1], sep='\t', index_col=0)
    spatial_data_2 = pd.read_csv(spatial_paths[1], index_col=0)

    extra_indices = spatial_data_2.index.difference(expression_data_2.index)
    spatial_data_2 = spatial_data_2.drop(extra_indices)

    # 3. Load the data for the third slice
    expression_data_3 = pd.read_csv(expression_paths[2], sep='\t', index_col=0)
    spatial_data_3 = pd.read_csv(spatial_paths[2], index_col=0)

    # 4. Combine the gene expression data for all three slices
    combined_expression = pd.concat([expression_data_1, expression_data_2, expression_data_3], axis=0)

    # 5. Create an AnnData object and preprocess
    adata = sc.AnnData(combined_expression)
    adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.01)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.pca(adata, n_comps=30)

    # 6. Perform clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    adata.obs['cluster'] = kmeans.fit_predict(adata.obsm['X_pca'])

    # 7. Assign clustering results to the first slice
    spatial_data_1['cluster'] = adata.obs['cluster'].iloc[:len(expression_data_1)].values
    spatial_data_1.to_csv(output_paths[0])

    # 8. Assign clustering results to the second slice
    start_idx_2 = len(expression_data_1)
    end_idx_2 = start_idx_2 + len(expression_data_2)
    spatial_data_2['cluster'] = adata.obs['cluster'].iloc[start_idx_2:end_idx_2].values
    spatial_data_2.to_csv(output_paths[1])

    # 9. Assign clustering results to the third slice
    start_idx_3 = end_idx_2
    spatial_data_3['cluster'] = adata.obs['cluster'].iloc[start_idx_3:].values
    spatial_data_3.to_csv(output_paths[2])


def load_data(expression_path, annotation_path):
    """
    Load gene expression data and cell type annotation data from CSV files.

    Parameters:
    - expression_path: Path to the gene expression CSV file.
    - annotation_path: Path to the cell type annotation CSV file.

    Returns:
    - expression_data: DataFrame with gene expression data.
    - annotations: Series with cell type annotations.
    """
    # Load gene expression data
    expression_data = pd.read_csv(expression_path, index_col=0)

    # Load annotations (assumed to be in a separate file)
    annotations = pd.read_csv(annotation_path, index_col=0)[
        'cell_type']  # Replace 'cell_type' with the actual column name if needed

    return expression_data, annotations


def calculate_differential_genes(expression_data, annotations, n_top_genes=50):
    """
    Calculate differential expression genes for each cell type compared to all other cell types.

    Parameters:
    - expression_data: DataFrame with gene expression data.
    - annotations: Series with cell type annotations.
    - n_top_genes: The number of top differentially expressed genes to return for each cell type.

    Returns:
    - results_dict: Dictionary with cell types as keys and DataFrame of differential genes as values.
    """
    results_dict = {}

    # Get the unique cell types
    cell_types = annotations.unique()

    # Loop through each cell type and calculate differential expression
    for cell_type in cell_types:
        # Select the cells of the current cell type and the others
        cell_type_cells = annotations == cell_type
        other_cells = annotations != cell_type

        # Extract the expression data for the current cell type and the other cell types
        expr_cell_type = expression_data.loc[cell_type_cells]
        expr_other_cells = expression_data.loc[other_cells]

        # Calculate the log fold change and p-value for each gene
        log_fold_changes = np.log2(expr_cell_type.mean(axis=0) / expr_other_cells.mean(axis=0))
        pvals = [stats.ttest_ind(expr_cell_type[gene], expr_other_cells[gene])[1] for gene in expression_data.columns]

        # Create a DataFrame with the results
        df_results = pd.DataFrame({
            'gene': expression_data.columns,
            'logfoldchanges': log_fold_changes,
            'pvals': pvals
        })

        # Adjust p-values using the Benjamini-Hochberg method
        df_results['pvals_adj'] = pd.Series(stats.models.multipletests(df_results['pvals'], method='fdr_bh')[1])

        # Sort the results by log fold change (descending) and p-value (ascending)
        df_results_sorted = df_results.sort_values(by=['logfoldchanges', 'pvals'], ascending=[False, True])

        # Store the top n genes in the results dictionary
        results_dict[cell_type] = df_results_sorted.head(n_top_genes)

    return results_dict

