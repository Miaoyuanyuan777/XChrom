## # function to transform scRNA-seq/scATAC-seq data to h5ad file
import anndata
from pathlib import Path
from typing import Union

## scRNA-seq: mtx,seurat(rds,h5seurat),.loom(anndata def read_loom())
## scATAC-seq: h5
## paired-scRNA-seq/scATAC-seq: h5

def read_10x_h5_to_h5ad(
        h5_file: Union[str, Path],
        output_file: Union[str, Path],
        genome: str = None,
        paired: bool = False,
        ) -> anndata.AnnData:
    """
    Read 10x Genomics h5 file and convert to h5ad format
    
    Parameters
    ----------
    h5_file:
        Path to 10x Genomics h5 file
    output_file:
        Output h5ad file path
    genome:
        Name of the genome. If None, uses the first available genome
    paired:
        Whether input paired RNA and ATAC adata objects, default False
        
    Returns
    -------
    anndata.AnnData
        Converted AnnData object
    """
    import scanpy as sc
    
    h5_file = Path(h5_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read 10x h5 file using scanpy
    adata = sc.read_10x_h5(
        filename=str(h5_file),
        genome=genome,
        gex_only=False
    )
    
    # Make variable names unique (in case of duplicate gene names)
    adata.var_names_make_unique()
    
    # Add some basic information
    adata.var['gene_ids'] = adata.var.index
    if 'gene_symbols' in adata.var.columns:
        adata.var['gene_symbols'] = adata.var['gene_symbols']
    # Save to h5ad format
    adata.write_h5ad(output_file)
    print(f"Successfully converted {h5_file} to {output_file}")
    print(f"Data shape: {adata.n_obs} cells × {adata.n_vars} features")
    
    if paired:
        ad_rna = adata[:, adata.var['feature_types']=='Gene Expression']
        ad_atac = adata[:, adata.var['feature_types']=='Peaks']
        features = ad_atac.var['gene_ids']
        chromosome_start_end = features.str.split(":", expand=True)
        chromosome_start_end.columns = ['chr', 'start_end']
        start_end = chromosome_start_end['start_end'].str.split("-", expand=True)
        chromosome_start_end['start'] = start_end[0]
        chromosome_start_end['end'] = start_end[1]

        ad_atac.var['chr'] = chromosome_start_end['chr']
        ad_atac.var['start'] = chromosome_start_end['start'].astype(int)
        ad_atac.var['end'] = chromosome_start_end['end'].astype(int)
        out_str = str(output_file)
        atac_path = out_str.replace('.h5ad', '_atac.h5ad')
        rna_path = out_str.replace('.h5ad', '_rna.h5ad')
        ad_atac.write_h5ad(atac_path)
        print(f"Successfully extracted ATAC data from {output_file} to {atac_path}")
        print(f"Data shape: {ad_atac.n_obs} cells × {ad_atac.n_vars} peaks")
        ad_rna.write_h5ad(rna_path)
        print(f"Successfully extracted RNA data from {output_file} to {rna_path}")
        print(f"Data shape: {ad_rna.n_obs} cells × {ad_rna.n_vars} genes")
        return ad_rna, ad_atac
    else:
        return adata
    
