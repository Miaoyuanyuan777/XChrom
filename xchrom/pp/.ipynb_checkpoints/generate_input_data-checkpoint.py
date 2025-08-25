#!/usr/bin/env python

## if you want to generate train data for XChrom, please use process_train_test_single function directly.
## if you want to generate test data for XChrom, when the train set and test set are from single dataset, process_train_test_single will generate train and test inputs simultaneously,
# and when the train set and test set are from different datasets, please use process_train_test_dual function to generate test inputs.

import os
import h5py
import anndata
import configargparse
import numpy as np
import scipy.sparse as sparse
from pathlib import Path
from typing import Union
from ._utils import split_test, make_h5_sparse

def make_parser():
    parser = configargparse.ArgParser(
        description="Preprocess anndata to generate inputs for XChrom.")
    parser.add_argument('--ad_atac', type=str,
                       help='Input scATAC anndata. .var must have chr, start, end columns. anndata.X must be in csr format.')
    parser.add_argument('--input_fasta', type=str,
                       help='Genome fasta file.')
    parser.add_argument('--out_path', type=str, default='./train_data/',
                       help='Output path. Default to ./train_data/')
    parser.add_argument('--dual', type = bool, default = False,
                        help = 'Whether to generate train and test set from dual dataset.')
    return parser

def _prepare_anndata(ad_file, output_path):
    """prepare anndata and output directory"""
    ad_file = Path(ad_file)
    output_path = Path(output_path)
    ad = anndata.read_h5ad(ad_file)
    os.makedirs(output_path, exist_ok=True)
    
    # generate sequence depth
    Ln = np.log10(ad.obs['n_genes'])
    b = (Ln - np.mean(Ln)) / np.std(Ln)
    ad.obs['b_zscore'] = b
    
    return ad

def _save_basic_files(ad, output_path):
    """save basic files"""
    # save anndata
    output_path = Path(output_path)
    ad.write_h5ad(output_path /'ad.h5ad')
    print('train/test data is saved in: ', output_path)
    
    # save peak bed file
    ad.var.loc[:,['chr','start','end']].to_csv(output_path /'peaks.bed', sep='\t', header=False, index=False)
    print('successful writing bed file.')

def process_train_test_single(
    ad_atac: Union[str, Path, anndata.AnnData], 
    input_fasta: str, 
    output_path: str = './train_data/'
    ):
    """
    Generate XChrom training and test inputs from a single dataset.
    
    Parameters
    ----------
    ad_atac : str or Path
        scATAC anndata file path, need to be processed by scanpy's filter_genes and filter_cells functions, get .obs['n_genes']
    input_fasta : str or Path
        genome fasta file path
    output_path : str or Path, optional
        output path, default is './train_data/'
        
    Returns
    -------
    dict
    A dictionary containing the following keys:
        'anndata': The original atac anndata object,
        'trainval_cell_index': The indices of cells in the train/val set
        'test_cell_index': The indices of cells in the test set,
        'trainval_peak_index': The indices of peaks in the train/val set,
        'test_peak_index': The indices of peaks in the test set,
    """
    input_fasta = Path(input_fasta)
    output_path = Path(output_path)
    if isinstance(ad_atac, str) or isinstance(ad_atac, Path):
        ad = _prepare_anndata(ad_atac, output_path)
    elif isinstance(ad_atac, anndata.AnnData):
        ad = ad_atac
        os.makedirs(output_path, exist_ok=True)
        # generate sequence depth
        Ln = np.log10(ad.obs['n_genes'])
        b = (Ln - np.mean(Ln)) / np.std(Ln)
        ad.obs['b_zscore'] = b
    else:
        raise ValueError('ad_atac must be a str, Path, or anndata.AnnData object')
        
    _save_basic_files(ad, output_path)

    # data split
    trainval_cell, test_cell, trainval_peak, test_peak = split_test(
        np.arange(ad.shape[0]), np.arange(ad.shape[1]))
    
    # save data split
    f = h5py.File(output_path /'splits.h5', "w")
    f.create_dataset("trainval_cell", data=trainval_cell)
    f.create_dataset("trainval_peak", data=trainval_peak)
    f.create_dataset("test_cell", data=test_cell)
    f.create_dataset('test_peak', data=test_peak)
    f.close()
    print('successful writing train/test split file.')
    
    # save split anndata
    ad_trainval = ad[trainval_cell, trainval_peak]
    ad_test = ad[test_cell, test_peak]
    ad_crosscell = ad[test_cell, trainval_peak]
    ad_crosspeak = ad[trainval_cell, test_peak]
    
    ad_trainval.write_h5ad(output_path /'ad_trainval.h5ad')
    ad_test.write_h5ad(output_path /'ad_test.h5ad')
    ad_crosscell.write_h5ad(output_path /'ad_crosscell.h5ad')
    ad_crosspeak.write_h5ad(output_path /'ad_crosspeak.h5ad')
    print('successful writing train/test anndata file.')

    # save other bed files
    ad_trainval.var.loc[:,['chr','start','end']].to_csv(output_path /'peaks_trainval.bed', sep='\t', header=False, index=False)
    ad_test.var.loc[:,['chr','start','end']].to_csv(output_path /'peaks_test.bed', sep='\t', header=False, index=False)

    # save label matrix
    m = ad.X.tocoo().transpose().tocsr()  # m: csr matrix, rows as seqs, cols are cells
    m_trainval = m[trainval_peak,:][:,trainval_cell]
    m_test = m[test_peak,:][:,test_cell]
    m_crosscell = m[trainval_peak,:][:,test_cell]
    m_crosspeak = m[test_peak,:][:,trainval_cell]
    
    sparse.save_npz(output_path /'m_trainval.npz', m_trainval, compressed=False)
    sparse.save_npz(output_path /'m_test.npz', m_test, compressed=False)
    sparse.save_npz(output_path /'m_crosscell.npz', m_crosscell, compressed=False)
    sparse.save_npz(output_path /'m_crosspeak.npz', m_crosspeak, compressed=False)
    print('successful writing sparse m.')

    # save sequence h5 file
    make_h5_sparse(ad, output_path /'all_seqs.h5', input_fasta)
    print('Successfully saving all sequence h5 file...')
    make_h5_sparse(ad_trainval, output_path /'trainval_seqs.h5', input_fasta)
    print('Successfully saving trainval sequence h5 file...')
    make_h5_sparse(ad_test, output_path /'test_seqs.h5', input_fasta)
    print('Successfully saving test sequence h5 file...')

    return {
        'anndata': ad,
        'trainval_cell_index': trainval_cell,
        'test_cell_index': test_cell,
        'trainval_peak_index': trainval_peak,
        'test_peak_index': test_peak
    }

def process_test_dual(
    ad_atac: Union[str, Path, anndata.AnnData], 
    input_fasta: Union[str, Path], 
    output_path: Union[str, Path] = './test_data/'
    ):
    """
    Generate XChrom train set and test set from 2 datasets, to generate XChrom test inputs.
    
    Parameters
    ----------
    ad_atac : str or Path or anndata.AnnData
        scATAC anndata file path, need to be processed by scanpy's filter_genes and filter_cells functions, get .obs['n_genes']
    input_fasta : str or Path
        genome fasta file path
    output_path : str or Path, optional
        output path, default is './test_data/'
        
    Returns
    -------
    ad: The original atac anndata object
    """
    if isinstance(ad_atac, str) or isinstance(ad_atac, Path):
        ad = _prepare_anndata(ad_atac, output_path)
    elif isinstance(ad_atac, anndata.AnnData):
        ad = ad_atac
        os.makedirs(output_path, exist_ok=True)
        # generate sequence depth
        Ln = np.log10(ad.obs['n_genes'])
        b = (Ln - np.mean(Ln)) / np.std(Ln)
        ad.obs['b_zscore'] = b
    else:
        raise ValueError('ad_atac must be a str, Path, or anndata.AnnData object')
    
    input_fasta = Path(input_fasta)
    output_path = Path(output_path)
    _save_basic_files(ad, output_path)

    # save label matrix
    m = ad.X.tocoo().transpose().tocsr()  # m: csr matrix, rows as seqs, cols are cells
    sparse.save_npz(output_path /'m.npz', m, compressed=False)
    print('successful writing sparse m.')

    # save sequence h5 file
    print('start saving all sequence h5 file...')
    make_h5_sparse(ad, output_path /'all_seqs.h5', input_fasta)

    return ad

def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.dual:
        process_test_dual(args.ad_file, args.input_fasta, args.out_path)
    else:
        process_train_test_single(args.ad_file, args.input_fasta, args.out_path)

if __name__ == "__main__":
    main()