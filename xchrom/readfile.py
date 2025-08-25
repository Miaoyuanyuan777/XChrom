## # function to transform scRNA-seq/scATAC-seq data to h5ad file
from typing_extensions import Literal
import anndata
import scanpy as sc
from pathlib import Path
from typing import Union

## scRNA-seq: mtx,seurat(rds,h5seurat),.loom(anndata def read_loom())
## scATAC-seq: h5
## paired-scRNA-seq/scATAC-seq: h5


def read_10x_mtx_to_adata(
    path: str,
    output_file: str = None,
    prefix: str = None,
    mtx_folder: str = 'filtered_feature_bc_matrix',
    **kwargs,
    ) -> sc.AnnData:
    """
    Read transcriptomics or chromatin accessibility data of 10X.
    
    Parameters
    ----------
    path:
        10x data root directory path
    output_file:
        Output h5ad file path, default None, if None, do not save to file
    prefix:
        Prefix for the input file
    mtx_folder:
        MTX format data folder name (default 'filtered_feature_bc_matrix')
    **kwargs:
        Additional arguments for sc.read_10x_mtx
    
    Returns
    -------
        anndata.Anndata object
    """
    path = Path(path)
    mtx_path = path / mtx_folder
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use MTX format
    if mtx_path.exists():
        adata = sc.read_10x_mtx(
            mtx_path,  # Path to directory for .mtx,.tsv,.mtx.gz,.tsv.gz
            var_names='gene_symbols',  # Use gene symbols as variable names
            make_unique=True,          # Ensure gene names are unique
            cache=True,                 # Cache read results
            prefix=prefix,              # Prefix before matrix.mtx,features.tsv,barcodes.tsv
            **kwargs,
        )
            
        if output_file:
                adata.write_h5ad(output_file)
        return adata
    raise FileNotFoundError(f"No valid 10x data file found in {path}")


def read_rds_to_h5ad(
        seurat_rds: Union[str, Path],
        output_file: Union[str, Path],
        assay: Literal['RNA', 'ATAC'] = 'RNA',
        slot: str = 'counts',
        ) -> anndata.AnnData:
    """
    Read Seurat object (.rds) to h5ad format
    
    Parameters
    ----------
    seurat_rds:
        Seurat RDS file path
    assay:
        Used assay (RNA or ATAC) (default 'RNA')
    slot:
        Used slot (default 'counts')
    output_file:
        Output h5ad file path
        
    """
    seurat_rds = Path(seurat_rds)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # R console preparation
    from rpy2.robjects import r
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    
    # R fucntion preparation
    readRDS = robjects.r['readRDS']
    to_array = robjects.r['as.array']
    transpose = robjects.r['t']

    # required R package loading

    Seurat = importr('Seurat')
    SeuratObject = importr('SeuratObject')

    # get data as single cell
    GetMetaData = r("""
    function(seurat_obj){
    library(Seurat)
    metadata=seurat_obj@meta.data
    return(metadata)
    }
    """)

    GetMetaFeature = r("""
    function(seurat_obj,assay){
    library(Seurat)
    library(dplyr)
    gene_feature=GetAssay(seurat_obj,assay=assay)@meta.features
    gene_feature$'genome'=1
    return(gene_feature)
    }
    """)

    seurat_obj = readRDS(seurat_rds)

    # raw count matrix
    count_mat = SeuratObject.GetAssayData(seurat_obj, assay=assay, slot=slot)
    count_mat = to_array(count_mat)
    count_mat = count_mat.transpose()

    # get gene names
    metafeature = GetMetaFeature(seurat_obj, assay)

    # metadata
    metadata = GetMetaData(seurat_obj)

    # create anndata
    adata = anndata.AnnData(X=count_mat, var=metafeature, obs=metadata)
    adata.write_h5ad(output_file)
    return adata


def read_h5seurat_to_h5ad(
        h5seurat_file: Union[str, Path],
        output_file: Union[str, Path],
        assay: Literal['RNA', 'ATAC'] = 'RNA',
        slot: str = 'counts',
        ) -> anndata.AnnData:
    """
    Read h5seurat file and convert to h5ad format
    
    Parameters
    ----------
    h5seurat_file:
        h5seurat file path
    assay:
        Used assay (RNA or ATAC) (default 'RNA')
    slot:
        Used slot (default 'counts')
    output_file:
        Output h5ad file path
        
    Returns
    -------
    anndata.AnnData
        Converted AnnData object
    """
    h5seurat_file = Path(h5seurat_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare R environment
    from rpy2.robjects import r
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    
    # Load required R packages
    Seurat = importr('Seurat')
    SeuratObject = importr('SeuratObject')
    
    # Define R function to load h5seurat file
    LoadH5Seurat = r("""
    function(h5_path) {
        library(Seurat)
        seurat_obj <- LoadH5Seurat(h5_path)
        return(seurat_obj)
    }
    """)
    
    # Define R function to get metadata
    GetMetaData = r("""
    function(seurat_obj){
        library(Seurat)
        metadata=seurat_obj@meta.data
        return(metadata)
    }
    """)

    # Define R function to get feature information
    GetMetaFeature = r("""
    function(seurat_obj, assay){
        library(Seurat)
        library(dplyr)
        gene_feature=GetAssay(seurat_obj, assay=assay)@meta.features
        gene_feature$'genome'=1
        return(gene_feature)
    }
    """)
    
    # Load h5seurat file
    seurat_obj = LoadH5Seurat(str(h5seurat_file))
    
    # Get count matrix
    count_mat = SeuratObject.GetAssayData(seurat_obj, assay=assay, slot=slot)
    count_mat = robjects.r['as.array'](count_mat)
    count_mat = count_mat.transpose()
    
    # Get gene names and metafeatures
    metafeature = GetMetaFeature(seurat_obj, assay)
    
    # Get metadata
    metadata = GetMetaData(seurat_obj)
    
    # Create AnnData object
    adata = anndata.AnnData(X=count_mat, var=metafeature, obs=metadata)
    adata.write_h5ad(output_file)
    return adata


def read_seurat_to_h5ad(
        seurat_file: Union[str, Path],
        output_file: Union[str, Path],
        assay: Literal['RNA', 'ATAC'] = 'RNA',
        slot: str = 'counts',
        ) -> anndata.AnnData:
    """
    Read Seurat object (RDS or h5seurat format) and convert to h5ad format
    
    Parameters
    ----------
    seurat_file:
        Path of Seurat object (RDS or h5seurat format)
    assay:
        Used assay (RNA or ATAC) (default 'RNA')
    slot:
        Used slot (default 'counts')
    output_file:
        Output h5ad file path
    
    Returns
    -------
    anndata.AnnData
        Converted AnnData object
    """
    file_path = Path(seurat_file)
    if file_path.suffix.lower() == '.rds':
        return read_rds_to_h5ad(seurat_file, output_file, assay, slot)
    elif file_path.suffix.lower() == '.h5seurat':
        return read_h5seurat_to_h5ad(seurat_file, output_file, assay, slot)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .rds, .h5seurat")


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
    
