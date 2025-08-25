import numpy as np
import anndata
import h5py
import time
from typing import Tuple
from pathlib import Path
from typing import Union, Literal
from .._utils import make_bed_seqs_from_df, dna_1hot_2vec

def split_test(
    cellids: np.ndarray,
    peakids: np.ndarray,
    seed: int = 20,
    ratio: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    split cellids and peakids into train/test sets
    
    Parameters
    ----------
    cellids: np.ndarray
        A 1D array of cell ids, to be split into train/test sets
    peakids: np.ndarray
        A 1D array of peak ids, to be split into train/test sets
    seed: int, default is 20
        random seed for reproducibility
    ratio: float, default is 0.9
        ratio of train set, the rest will be used for testing
        
    Returns
    -------
    trainval_cell: np.ndarray
        A 1D array of cell ids, to be used for training and validation
    test_cell: np.ndarray
        A 1D array of cell ids, to be used for testing
    trainval_peak: np.ndarray
        A 1D array of peak ids, to be used for training and validation
    test_peak: np.ndarray
        A 1D array of peak ids, to be used for testing
    """
    np.random.seed(seed=seed)
    test_cell = np.random.choice(cellids,int(len(cellids) * (1 - ratio)),replace=False)
    trainval_cell = np.setdiff1d(cellids, test_cell)
    test_peak = np.random.choice(peakids,int(len(peakids) * (1 - ratio)),replace=False)
    trainval_peak = np.setdiff1d(peakids, test_peak)
    return trainval_cell,test_cell,trainval_peak,test_peak

def make_h5_sparse(
    tmp_ad: anndata.AnnData, 
    h5_name: Union[str, Path], 
    input_fasta: Union[str, Path], 
    seq_len: int = 1344, 
    batch_size: int = 1000
    ):
    """
    save the sequence of peaks to h5 file,which is used to generate train/test data for XChrom model training
    
    Parameters
    ----------
    tmp_ad: anndata.AnnData
        anndata object, must have .var['chr','start','end']
    h5_name: Union[str, Path]
        name of the h5 file, which will be saved in the current directory
    input_fasta: Union[str, Path]
        path to the genome fasta file
    seq_len: int, default is 1344
        length of the sequence
    batch_size: int, default is 1000
        how many peaks to process at a time
    """
    h5_name = Path(h5_name)
    input_fasta = Path(input_fasta)
    t0 = time.time()
    n_peaks = tmp_ad.shape[1]
    bed_df = tmp_ad.var.loc[:,['chr','start','end']] # bed file
    bed_df.index = np.arange(bed_df.shape[0])
    # n_batch = int(np.floor(n_peaks/batch_size))
    n_batch = max(1, int(np.ceil(n_peaks / batch_size)))
    # batches = np.array_split(np.arange(n_peaks), n_batch) # split all peaks to process in batches
    batches = np.array_split(np.arange(n_peaks), n_batch) if n_peaks > 0 else []
    
    # create h5 file 
    # X is a matrix of n_peaks * 1344, which is the sequence of peaks
    f = h5py.File(h5_name, "w")
    ds_X = f.create_dataset("X",(n_peaks, seq_len),dtype="int8")

    # save to h5 file
    for i in range(len(batches)):
        idx = batches[i]
        # write X to h5 file
        seqs_dna,_ = make_bed_seqs_from_df(
            bed_df.iloc[idx,:],
            fasta_file=input_fasta,
            seq_len=seq_len,
        )
        dna_array_dense = [dna_1hot_2vec(x) for x in seqs_dna]
        dna_array_dense = np.array(dna_array_dense)
        ds_X[idx] = dna_array_dense
        t1 = time.time()
        total = t1-t0
        # print('process %d peaks takes %.1f s' %(i*batch_size, total))
    f.close()

def filter_multiome_data(
    ad_rna: anndata.AnnData,
    ad_atac: anndata.AnnData,
    filter_ratio: float = 0.05,
    species: Literal['mouse', 'human'] = 'human',
    min_genes: int = 0,
    min_cells: int = 0
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    Filter multiome RNA and ATAC data based on expression/accessibility thresholds and chromosomes
    
    Parameters
    ----------
    ad_rna: anndata.AnnData
        RNA expression data (cells x genes)
    ad_atac: anndata.AnnData  
        ATAC accessibility data (cells x peaks)
    filter_ratio: float, default is 0.05
        Minimum ratio of cells in which a gene/peak should be expressed/accessible
    species: str, default is 'mouse'
        Species name to determine chromosome filtering. Supports 'mouse' and 'human'
    min_genes: int, default is 0
        Minimum number of genes expressed per cell
    min_cells: int, default is 0
        Minimum number of cells expressing each gene
        
    Returns
    -------
    ad_rna_filtered: anndata.AnnData
        Filtered RNA data
    ad_atac_filtered: anndata.AnnData
        Filtered ATAC data
        
    Examples
    --------
    >>> import xchrom as xc
    >>> ad_rna, ad_atac = xc.pp.filter_multiome_data(
    ...     ad_rna=ad_rna,
    ...     ad_atac=ad_atac,
    ...     species='mouse',
    ...     filter_ratio=0.05,
    ...     min_genes=0,
    ...     min_cells=0
    ... )
    """
    import scanpy as sc
    
    # Make copies to avoid modifying original data
    ad_rna_filtered = ad_rna.copy()
    ad_atac_filtered = ad_atac.copy()
    
    # Basic filtering
    sc.pp.filter_cells(ad_rna_filtered, min_genes=min_genes)
    sc.pp.filter_genes(ad_rna_filtered, min_cells=min_cells)
    sc.pp.filter_cells(ad_atac_filtered, min_genes=min_genes)
    sc.pp.filter_genes(ad_atac_filtered, min_cells=min_cells)

    # Filter based on expression/accessibility ratio
    thres = int(ad_atac_filtered.shape[0] * filter_ratio)
    ad_rna_filtered = ad_rna_filtered[:, ad_rna_filtered.var['n_cells'] > thres]
    ad_atac_filtered = ad_atac_filtered[:, ad_atac_filtered.var['n_cells'] > thres]
    
    # Filter peaks by chromosome
    if species.lower() == 'mouse':
        # Mouse: chr1-chr19, chrX, chrY
        chrs = ['chr' + str(i) for i in range(1, 20)] + ['chrX', 'chrY']
    elif species.lower() == 'human':
        # Human: chr1-chr22, chrX, chrY
        chrs = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    else:
        raise ValueError(f"Unsupported species: {species}. Supported species are 'mouse' and 'human'")
    
    # Check if chromosome column exists in ATAC data
    if 'chr' in ad_atac_filtered.var.columns:
        ad_atac_filtered = ad_atac_filtered[:, ad_atac_filtered.var['chr'].isin(chrs)]
    else:
        print("Warning: 'chr' column not found in ATAC data. Skipping chromosome filtering.")
    
    print(f"RNA data after filtering: {ad_rna_filtered}")
    print(f"ATAC data after filtering: {ad_atac_filtered}")
    
    return ad_rna_filtered, ad_atac_filtered
