# evaluate model performance on cross-cell,cross-peak,cross-both prediction in within sample scenario

import scanpy as sc
import h5py
import numpy as np
import scipy.sparse as sparse
from typing import Union
import os
import anndata
from scipy import stats
from pathlib import Path
from ._utils import calc_auc_pr, calc_nsls_score, calc_pca
from ..tr._utils import Generator, XChrom_model
from ..pl._utils import plot_percell_aucprc, plot_perpeak_aucprc

def crosscell_aucprc(
    cell_embedding_ad:Union[str, Path, anndata.AnnData],
    input_folder:Union[str, Path] = './train_data',
    model_path:Union[str, Path] = './train_out/E1000best_model.h5',
    output_path:Union[str, Path] = './eval_out',
    cellembed_raw:str = 'X_pca',
    save_pred:bool = False,
    scatter_plot:bool = False
    ) -> dict:
    """
    Evaluate the performance in cross-cell prediction with within-sample data, calculate auROC & auPRC for overall, per-cell and per-peak.

    Parameters
    ----------
    cell_embedding_ad: str or Path or anndata.AnnData
        Path to the cell embeddings adata file. provide cell input embeddings for XChrom model prediction.
    input_folder: str or Path
        Path to the train data folder. Should generate by XChrom_preprocess.py. 
        Must contain 'splits.h5', 'ad_crosscell.h5ad', 'm_crosscell.npz', 'trainval_seqs.h5'.
    model_path: str or Path
        Path to the trained model.
    output_path: str or Path
        Path to the output folder.
    cellembed_raw: str
        Key of the raw cell embeddings in the cell embeddings adata.
    save_pred: bool
        Whether to save the prediction matrix with npy format.
    scatter_plot: bool
        Whether to plot the scatter plot of the per-cell & per peak auROC and auPRC.

    Returns
    -------
    dict
        Dictionary containing:
            'overall_auroc': overall auROC,
            'overall_auprc': overall auPRC,
            'percell_auroc': per-cell auROC,
            'percell_auprc': per-cell auPRC,
            'perpeak_auroc': per-peak auROC,
            'perpeak_auprc': per-peak auPRC.
    Examples
    --------
    >>> import xchrom as xc
    metrics1 = xc.tl.crosscell_aucprc(
        cell_embedding_ad='./data/1_within_sample/m_brain_paired_rna.h5ad',
        input_folder='./data/1_within_sample/train_data',
        model_path='./data/1_within_sample/train_out/E1000best_model.h5',
        output_path='./data/1_within_sample/eval_out',
        cellembed_raw='X_pca',
        save_pred=True,
        scatter_plot=True
        )
    """

    if isinstance(cell_embedding_ad, str) or isinstance(cell_embedding_ad, Path):
        cell_embedding_ad = Path(cell_embedding_ad)
        rna = sc.read_h5ad(cell_embedding_ad)
    elif isinstance(cell_embedding_ad, anndata.AnnData):
        rna = cell_embedding_ad
    else:
        raise ValueError(f"cell_embedding_ad must be a str, Path or anndata.AnnData!")
    input_folder = Path(input_folder)
    model_path = Path(model_path)
    output_path = Path(output_path)
    cellembed_raw = str(cellembed_raw)

    if not input_folder.exists():
        raise FileNotFoundError(f"Data folder {input_folder} not found! Run XChrom_preprocess.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model path {model_path} not found! Run XChrom_train.py first.")
    if not cell_embedding_ad.exists():
        raise FileNotFoundError(f"Cell embedding adata path {cell_embedding_ad} not found! Please check the path.")
    os.makedirs(output_path, exist_ok=True)
    if cellembed_raw not in rna.obsm:
        raise ValueError(f"Cell embedding key {cellembed_raw} not found in the cell embedding adata!")
    rna.obsm['zscore32_perpc'] = stats.zscore(np.array(rna.obsm[cellembed_raw]), axis=0) 
        # Actually, after model training, the rna local file will be updated, adding the cell_input_key, and the zscore does not need to be recalculated.
    # if cell_input_key not in rna.obsm:
    #     rna.obsm[cell_input_key] = stats.zscore(np.array(rna.obsm['X_pca']), axis=0)   
    with h5py.File(input_folder/'splits.h5', 'r') as hf:
        test_cellid = hf['test_cell'][:]   
    rna_crosscell = rna[test_cellid,:]
    atac_crosscell = sc.read_h5ad(input_folder/'ad_crosscell.h5ad')
    assert atac_crosscell.obs.index.equals(rna_crosscell.obs.index), "Indexes differ!"
    atac_crosscell.obsm['zscore32_perpc'] = rna_crosscell.obsm['zscore32_perpc']
    m_crosscell = sparse.load_npz(input_folder/'m_crosscell.npz').tocsr() 

    crosscell_gen = Generator(
        seq_path = input_folder/'trainval_seqs.h5',
        adata = atac_crosscell,
        cell_input_key = 'zscore32_perpc',
        m = m_crosscell
    )
    test_ds = crosscell_gen.create_dataset(shuffle=False)
    model = XChrom_model(n_cells=atac_crosscell.shape[0],show_summary=False)
    model.load_weights(model_path)

    pred = model.predict(test_ds)
    print('predction shape:', pred.shape)

    m_crosscell_to1 = m_crosscell.copy()  
    m_crosscell_to1[m_crosscell_to1 != 0] = 1  # binaries
    test_prediction = pred.T
    true_01matrix = m_crosscell_to1.T.toarray()

    ## calculate auc and pr
    ## -1 Calculate overall auROC & auPRC 
    overall_metrics = calc_auc_pr(true_01matrix, test_prediction, 'overall')
    print(f"Overall auROC: {overall_metrics['auroc']:.4f}, auPRC: {overall_metrics['auprc']:.4f}")

    ## -2 Calculate per cell auRPC & auPRC
    percell_metrics = calc_auc_pr(true_01matrix, test_prediction, 'percell')
    print(f"Per-cell auROC: {percell_metrics['auroc']:.4f}, auPRC: {percell_metrics['auprc']:.4f}")
    print(f"Valid cells: {percell_metrics['n_cells']}")

    ## -3 Calculate per peak auROC & auPRC
    perpeak_metrics = calc_auc_pr(true_01matrix, test_prediction, 'perpeak')
    print(f"Per-peak auROC: {perpeak_metrics['auroc']:.4f}, auPRC: {perpeak_metrics['auprc']:.4f}")
    print(f"Valid peaks: {perpeak_metrics['n_peaks']}")

    if scatter_plot:
        plot_percell_aucprc(true_01matrix, test_prediction, out_file=output_path/'crosscell_percell_aucprc_scatterplot.pdf')
        plot_perpeak_aucprc(true_01matrix, test_prediction, out_file=output_path/'crosscell_perpeak_aucprc_scatterplot.pdf')
    if save_pred:
        np.save(output_path/'crosscell_pred.npy', test_prediction)

    return {
        'overall_auroc': overall_metrics['auroc'],
        'overall_auprc': overall_metrics['auprc'],
        'percell_auroc': percell_metrics['auroc'],
        'percell_auprc': percell_metrics['auprc'],
        'perpeak_auroc': perpeak_metrics['auroc'],
        'perpeak_auprc': perpeak_metrics['auprc']
    }

def crosscell_nsls(
    cell_embedding_ad:Union[str, Path, anndata.AnnData],
    input_folder:Union[str, Path] = './train_data',
    model_path:Union[str, Path] = './train_out/E1000best_model.h5',
    output_path:Union[str, Path] = './eval_out',
    cellembed_raw:str = 'X_pca',
    celltype:str = 'celltype',
    save_pred:bool = False,
    plot_umap:bool = False
    ):
    """
    Evaluate the performance in cross-cell prediction with within-sample data, calculate neighbor score and label score for test cells.
    Predict all data (excluding cross-peak peaks), then extract test cells to calculate nsls
    
    Parameters
    ----------
    cell_embedding_ad: str or Path or anndata.AnnData
        Path to the cell embeddings adata file. provide cell input embeddings for XChrom model prediction.
    input_folder: str or Path
        Path to the train data folder. Should generate by XChrom_preprocess.py. 
        Must contain 'splits.h5', 'ad_crosscell.h5ad', 'm_crosscell.npz', 'trainval_seqs.h5'.
    model_path: str or Path
        Path to the trained model.
    output_path: str or Path
        Path to the output folder.
    cellembed_raw: str
        Key of the raw cell embeddings in the cell embeddings adata,to calculate RNA neighbors.
    celltype: str
        Key of the cell type in the cell embeddings adata.
    save_pred: bool
        Whether to save the prediction matrix with h5ad format.
    plot_umap: bool
        Whether to plot the UMAP of the test cells.

    Returns
    -------
    metrics: dict
        Dictionary containing neighbor score(k=10,50,100) and label score(k=10,50,100) for test cells.
    
    Examples
    --------
    >>> import xchrom as xc
    metrics4 = xc.tl.crosscell_nsls(
        cell_embedding_ad='./data/1_within_sample/m_brain_paired_rna.h5ad',
        input_folder='./data/1_within_sample/train_data',
        model_path='./data/1_within_sample/train_out/E1000best_model.h5',
        output_path='./data/1_within_sample/eval_out',
        cellembed_raw='X_pca',
        celltype='pc32_leiden',
        save_pred=True,
        plot_umap=True
        )
    """
    import matplotlib.pyplot as plt
    
    if isinstance(cell_embedding_ad, str) or isinstance(cell_embedding_ad, Path):
        cell_embedding_ad = Path(cell_embedding_ad)
        rna = sc.read_h5ad(cell_embedding_ad)
    elif isinstance(cell_embedding_ad, anndata.AnnData):
        rna = cell_embedding_ad
    else:
        raise ValueError(f"cell_embedding_ad must be a str, Path or anndata.AnnData!")
    input_folder = Path(input_folder)
    model_path = Path(model_path)
    output_path = Path(output_path)
    cellembed_raw = str(cellembed_raw)
    celltype = str(celltype)

    if not input_folder.exists():
        raise FileNotFoundError(f"Data folder {input_folder} not found! Run XChrom_preprocess.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model path {model_path} not found! Run XChrom_train.py first.")
    if not cell_embedding_ad.exists():
        raise FileNotFoundError(f"Cell embedding adata path {cell_embedding_ad} not found! Please check the path.")
    os.makedirs(output_path, exist_ok=True)
    if cellembed_raw not in rna.obsm:
        raise ValueError(f"Cell embedding key {cellembed_raw} not found in the cell embedding adata!")
    rna.obsm['zscore32_perpc'] = stats.zscore(np.array(rna.obsm[cellembed_raw]), axis=0) 
    # Actually, after model training, the rna local file will be updated, adding the cell_input_key, and the zscore does not need to be recalculated.
    # if cell_input_key not in rna.obsm:
    #     rna.obsm[cell_input_key] = stats.zscore(np.array(rna.obsm[cellembed_raw]), axis=0) 

    with h5py.File(input_folder/'splits.h5', 'r') as hf:
        test_cellid = hf['test_cell'][:]
        trainval_peakid = hf['trainval_peak'][:] 
    ad0 = anndata.read_h5ad(f'{input_folder}/ad.h5ad')  ## peaks × cells
    assert ad0.obs.index.equals(rna.obs.index), "Indexes differ!"
    ad0.obsm['zscore32_perpc'] = rna.obsm['zscore32_perpc']
    ad0.obs[celltype]=rna.obs[celltype]
    ad0.obs['b_zscore'] = np.full(ad0.shape[0],1)

    ad = ad0[:,trainval_peakid] ## training peaks × cells

    gen = Generator(
        seq_path = input_folder/'all_seqs.h5',
        adata = ad,
        cell_input_key = 'zscore32_perpc'
    )
    ds = gen.create_dataset(shuffle=False)
    model = XChrom_model(n_cells=ad.shape[0],show_summary=False)
    model.load_weights(model_path)
    pred = model.predict(ds)

    adp = ad.copy()
    adp.X = pred.transpose(1,0)

    ## XChrom impute 
    ad1 = adp.copy()
    ad1 = calc_pca(ad1)  ## return ['X_pca'] in ad1.obsm
    ns100,ls100= calc_nsls_score(rna,ad1,100,celltype,test_cells =test_cellid,use_rep_rna = cellembed_raw,use_rep_atac='X_pca')
    print(f'neighbor score(100)={ns100:.4f},label score(100)={ls100:.4f}')

    ns50,ls50= calc_nsls_score(rna,ad1,50,celltype,test_cells =test_cellid,use_rep_rna = cellembed_raw,use_rep_atac='X_pca')
    print(f'neighbor score(50)={ns50:.4f},label score(50)={ls50:.4f}')

    ns10,ls10= calc_nsls_score(rna,ad1,10,celltype,test_cells =test_cellid,use_rep_rna = cellembed_raw,use_rep_atac='X_pca')
    print(f'neighbor score(10)={ns10:.4f},label score(10)={ls10:.4f}')

    if plot_umap:
        size_vector = [60 if idx in test_cellid else 10 for idx in range(ad1.n_obs)]
        alpha_vector = [1 if idx in test_cellid else 0.3 for idx in range(ad1.n_obs)]
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        sc.pl.umap(ad1, color=celltype, size=size_vector, alpha=alpha_vector, ax=ax, show=False)
        plt.savefig(output_path/'crosscell_umap.pdf', format='pdf', dpi=300)
    if save_pred:
        ad1.write_h5ad(output_path/'crosscell_impute.h5ad')
    return {
        'ns100': ns100,
        'ls100': ls100,
        'ns50': ns50,
        'ls50': ls50,
        'ns10': ns10,
        'ls10': ls10
    }

def crosspeak_aucprc(
    cell_embedding_ad:Union[str, Path, anndata.AnnData],
    input_folder:Union[str, Path] = './train_data',
    model_path:Union[str, Path] = './train_out/E1000best_model.h5',
    output_path:Union[str, Path] = './eval_out',
    cellembed_raw:str = 'X_pca',
    save_pred:bool = False,
    scatter_plot:bool = False,
    ):
    """
    Evaluate the performance in cross-peak prediction with within-sample data, calculate auROC & auPRC for overall, per-cell and per-peak.
    
    Parameters
    ----------
    cell_embedding_ad: str or Path or anndata.AnnData
        Path to the cell embedding adata file. provide cell input embedding for XChrom model prediction.
    input_folder: str or Path
        Path to the train data folder. Should generate by XChrom_preprocess.py. 
        Must contain 'splits.h5', 'ad_crosspeak.h5ad', 'm_crosspeak.npz', 'test_seqs.h5'.
    model_path: str or Path
        Path to the trained model.
    output_path: str or Path
        Path to the output folder.
    cellembed_raw:str
        Key of the raw cell embeddings in the cell embedding adata.
    save_pred: bool
        Whether to save the prediction matrix with npy format.
    scatter_plot: bool
        Whether to plot the scatter plot of the per-cell & per peak auROC and auPRC.

    Returns
    -------
    metrics: dict
        Dictionary containing overall auROC, per-cell auROC, per-peak auROC, overall auPRC, per-cell auPRC, per-peak auPRC.
    
    Examples
    --------
    >>> import xchrom as xc
    metrics2 = xc.tl.crosspeak_aucprc(
        cell_embedding_ad='./data/1_within_sample/m_brain_paired_rna.h5ad',
        input_folder='./data/1_within_sample/train_data',
        model_path='./data/1_within_sample/train_out/E1000best_model.h5',
        output_path='./data/1_within_sample/eval_out',
        cellembed_raw = 'X_pca',
        save_pred=True,
        scatter_plot=True
        )
    """
    
    if isinstance(cell_embedding_ad, str) or isinstance(cell_embedding_ad, Path):
        cell_embedding_ad = Path(cell_embedding_ad)
        rna = sc.read_h5ad(cell_embedding_ad)
    elif isinstance(cell_embedding_ad, anndata.AnnData):
        rna = cell_embedding_ad
    else:
        raise ValueError(f"cell_embedding_ad must be a str, Path or anndata.AnnData!")
    input_folder = Path(input_folder)
    model_path = Path(model_path)
    cellembed_raw = str(cellembed_raw)
    output_path = Path(output_path)

    if not input_folder.exists():
        raise FileNotFoundError(f"Data folder {input_folder} not found! Run XChrom_preprocess.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model path {model_path} not found! Run XChrom_train.py first.")
    if not cell_embedding_ad.exists():
        raise FileNotFoundError(f"Cell embedding adata path {cell_embedding_ad} not found! Please check the path.")
    os.makedirs(output_path, exist_ok=True)
    if cellembed_raw not in rna.obsm:
        raise ValueError(f"Cell embeddings key {cellembed_raw} not found in the cell embeddings adata!")
    rna.obsm['zscore32_perpc'] = stats.zscore(np.array(rna.obsm[cellembed_raw]), axis=0) 
        # Actually, after model training, the rna local file will be updated, adding the cell_input_key, and the zscore does not need to be recalculated.
    with h5py.File(input_folder/'splits.h5', 'r') as hf:
        trainval_cellid = hf['trainval_cell'][:]
    rna_crosspeak = rna[trainval_cellid,:]
    ad_crosspeak = anndata.read_h5ad(input_folder/'ad_crosspeak.h5ad')  
    assert ad_crosspeak.obs.index.equals(rna_crosspeak.obs.index), "Indexes differ!"
    ad_crosspeak.obsm['zscore32_perpc'] = rna_crosspeak.obsm['zscore32_perpc']
    m_crosspeak = sparse.load_npz(input_folder/'m_crosspeak.npz').tocsr()  

    gen = Generator(
        seq_path = input_folder/'test_seqs.h5',
        adata = ad_crosspeak,
        cell_input_key = 'zscore32_perpc',
        m = m_crosspeak
    )
    test_ds = gen.create_dataset(shuffle=False)
    model = XChrom_model(n_cells=ad_crosspeak.shape[0],show_summary=False)
    model.load_weights(model_path)
    pred = model.predict(test_ds)
    print('predction shape:', pred.shape)
    m_crosspeak_to1 = m_crosspeak.copy()  
    m_crosspeak_to1[m_crosspeak_to1 != 0] = 1 
    test_prediction = pred.T
    true_01matrix = m_crosspeak_to1.T.toarray()
    
    ## calculate auc and pr
    ## -1 Calculate overall auROC & auPRC 
    overall_metrics = calc_auc_pr(true_01matrix, test_prediction, 'overall')
    print(f"Overall auROC: {overall_metrics['auroc']:.4f}, auPRC: {overall_metrics['auprc']:.4f}")
    
    ## -2 Calculate per cell auRPC & auPRC
    percell_metrics = calc_auc_pr(true_01matrix, test_prediction, 'percell')
    print(f"Per-cell auROC: {percell_metrics['auroc']:.4f}, auPRC: {percell_metrics['auprc']:.4f}")
    print(f"Valid cells: {percell_metrics['n_cells']}")

    ## -3 Calculate per peak auROC & auPRC
    perpeak_metrics = calc_auc_pr(true_01matrix, test_prediction, 'perpeak')
    print(f"Per-peak auROC: {perpeak_metrics['auroc']:.4f}, auPRC: {perpeak_metrics['auprc']:.4f}")
    print(f"Valid peaks: {perpeak_metrics['n_peaks']}")
    if scatter_plot:
        plot_percell_aucprc(true_01matrix, test_prediction, out_file=output_path/'crosspeak_percell_aucprc_scatterplot.pdf')
        plot_perpeak_aucprc(true_01matrix, test_prediction, out_file=output_path/'crosspeak_perpeak_aucprc_scatterplot.pdf')
    if save_pred:
        np.save(output_path/'crosspeak_pred.npy', test_prediction)

    return {
        'overall_auroc': overall_metrics['auroc'],
        'overall_auprc': overall_metrics['auprc'],
        'percell_auroc': percell_metrics['auroc'],
        'percell_auprc': percell_metrics['auprc'],
        'perpeak_auroc': perpeak_metrics['auroc'],
        'perpeak_auprc': perpeak_metrics['auprc']
    }

def crossboth_aucprc(
    cell_embedding_ad:Union[str, Path, anndata.AnnData],
    input_folder:Union[str, Path] = './train_data',
    model_path:Union[str, Path] = './train_out/E1000best_model.h5',
    output_path:Union[str, Path] = './eval_out',
    cellembed_raw:str = 'X_pca',
    save_pred:bool = False,
    scatter_plot:bool = False
    ): 
    """
    Evaluate the performance in cross-both prediction with within-sample data, calculate auROC & auPRC for overall, per-cell and per-peak.
    
    Parameters
    ----------
    cell_embedding_ad: str or Path or anndata.AnnData
        Path to the cell embedding adata file. provide cell input embedding for XChrom model prediction.
    input_folder: str or Path
        Path to the train data folder. Should generate by XChrom_preprocess.py. 
        Must contain 'splits.h5', 'ad_test.h5ad', 'm_test.npz', 'test_seqs.h5'.
    model_path: str or Path
        Path to the trained model.
    output_path: str or Path
        Path to the output folder.
    cellembed_raw: str
        Key of the raw cell embeddings in the cell embeddings adata.
    save_pred: bool
        Whether to save the prediction matrix with npy format.
    scatter_plot: bool
        Whether to plot the scatter plot of the per-cell & per peak auROC and auPRC.

    Returns
    -------
    metrics: dict
        Dictionary containing overall auROC, per-cell auROC, per-peak auROC, overall auPRC, per-cell auPRC, per-peak auPRC.
    
    Examples
    --------
    >>> import xchrom as xc
    metrics3 = xc.tl.crossboth_aucprc(
        cell_embedding_ad='./data/1_within_sample/m_brain_paired_rna.h5ad',
        input_folder='./data/1_within_sample/train_data',
        model_path='./data/1_within_sample/train_out/E1000best_model.h5',
        output_path='./data/1_within_sample/eval_out',
        cellembed_raw='X_pca',
        save_pred=True,
        scatter_plot=True
        )
    """
    
    if isinstance(cell_embedding_ad, str) or isinstance(cell_embedding_ad, Path):
        cell_embedding_ad = Path(cell_embedding_ad)
        rna = sc.read_h5ad(cell_embedding_ad)
    elif isinstance(cell_embedding_ad, anndata.AnnData):
        rna = cell_embedding_ad
    else:
        raise ValueError(f"cell_embedding_ad must be a str, Path or anndata.AnnData!")
    input_folder = Path(input_folder)
    output_path = Path(output_path)
    model_path = Path(model_path)
    cellembed_raw = str(cellembed_raw)

    if not input_folder.exists():
        raise FileNotFoundError(f"Data folder {input_folder} not found! Run XChrom_preprocess.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model path {model_path} not found! Run XChrom_train.py first.")
    if not cell_embedding_ad.exists():
        raise FileNotFoundError(f"Cell embedding adata path {cell_embedding_ad} not found! Please check the path.")
    os.makedirs(output_path, exist_ok=True)
    if cellembed_raw not in rna.obsm:
        raise ValueError(f"Cell embeddings key {cellembed_raw} not found in the cell embeddings adata!")
    rna.obsm['zscore32_perpc'] = stats.zscore(np.array(rna.obsm[cellembed_raw]), axis=0) 
        # Actually, after model training, the rna local file will be updated, adding the cell_input_key, and the zscore does not need to be recalculated.
    with h5py.File(input_folder/'splits.h5', 'r') as hf:
        test_cellid = hf['test_cell'][:]
    rna_crossboth = rna[test_cellid,:]
    ad_crossboth = anndata.read_h5ad(input_folder/'ad_test.h5ad')
    assert ad_crossboth.obs.index.equals(rna_crossboth.obs.index), "Indexes differ!"
    ad_crossboth.obsm['zscore32_perpc'] = rna_crossboth.obsm['zscore32_perpc']
    m_crossboth = sparse.load_npz(input_folder/'m_test.npz').tocsr()

    gen = Generator(
        seq_path = input_folder/'test_seqs.h5',
        adata = ad_crossboth,
        cell_input_key = 'zscore32_perpc',
        m = m_crossboth
    )
    test_ds = gen.create_dataset(shuffle=False)
    model = XChrom_model(n_cells=ad_crossboth.shape[0],show_summary=False)
    model.load_weights(model_path)
    pred = model.predict(test_ds)
    print('predction shape:', pred.shape)
    m_crossboth_to1 = m_crossboth.copy()  
    m_crossboth_to1[m_crossboth_to1 != 0] = 1 
    test_prediction = pred.T
    true_01matrix = m_crossboth_to1.T.toarray()

    ## calculate auc and pr
    ## -1 Calculate overall auROC & auPRC 
    overall_metrics = calc_auc_pr(true_01matrix, test_prediction, 'overall')
    print(f"Overall auROC: {overall_metrics['auroc']:.4f}, auPRC: {overall_metrics['auprc']:.4f}")
    
    ## -2 Calculate per cell auRPC & auPRC
    percell_metrics = calc_auc_pr(true_01matrix, test_prediction, 'percell')
    print(f"Per-cell auROC: {percell_metrics['auroc']:.4f}, auPRC: {percell_metrics['auprc']:.4f}")
    print(f"Valid cells: {percell_metrics['n_cells']}")

    ## -3 Calculate per peak auROC & auPRC
    perpeak_metrics = calc_auc_pr(true_01matrix, test_prediction, 'perpeak')
    print(f"Per-peak auROC: {perpeak_metrics['auroc']:.4f}, auPRC: {perpeak_metrics['auprc']:.4f}")
    print(f"Valid peaks: {perpeak_metrics['n_peaks']}")
    
    if scatter_plot:
        plot_percell_aucprc(true_01matrix, test_prediction, out_file=output_path/'crossboth_percell_aucprc_scatterplot.pdf')
        plot_perpeak_aucprc(true_01matrix, test_prediction, out_file=output_path/'crossboth_perpeak_aucprc_scatterplot.pdf')
    if save_pred:
        np.save(output_path/'crossboth_pred.npy', test_prediction)

    return {
        'overall_auroc': overall_metrics['auroc'],
        'overall_auprc': overall_metrics['auprc'],
        'percell_auroc': percell_metrics['auroc'],
        'percell_auprc': percell_metrics['auprc'],
        'perpeak_auroc': perpeak_metrics['auroc'],
        'perpeak_auprc': perpeak_metrics['auprc']
    }
    
def denoise_nsls(
    cell_embedding_ad:Union[str, Path, anndata.AnnData],
    input_folder:Union[str, Path] = './train_data',
    output_path:Union[str, Path] = './eval_out',
    model_path:Union[str, Path] = './train_out/E1000best_model.h5',
    cellembed_raw:str = 'X_pca',
    celltype:str = 'celltype',
    save_pred:bool = False,
    plot_umap:bool = False
    ):
    """
    Evaluate the performance of denoise in within-sample data, calculate neighbor score(k=10,50,100) and label score(k=10,50,100) for all cells.
    
    Parameters
    ----------
    cell_embedding_ad: str or Path or anndata.AnnData
        Path to the cell embedding adata file. provide cell input embedding for XChrom model prediction.
    input_folder: str or Path
        Path to the train & test data folder. Should generate by XChrom_preprocess.py. 
        Must contain 'splits.h5', 'ad.h5ad', 'all_seqs.h5'.
    output_path: str or Path
        Path to the output folder.
    model_path: str or Path
        Path to the trained model.
    cellembed_raw: str
        Key of the raw cell input embedding in the cell embedding adata,to calculate RNA neighbors.
    save_pred: bool 
        Whether to save the prediction matrix with npy format.
    plot_umap: bool
        Whether to plot the UMAP of the test cells.

    Returns
    -------
    metrics: dict
        Dictionary containing neighbor score(k=10,50,100) and label score(k=10,50,100) for all cells.
    
    Examples
    --------
    >>> import xchrom as xc
    metrics5 = xc.tl.denoise_nsls(
        cell_embedding_ad='./data/1_within_sample/m_brain_paired_rna.h5ad',
        input_folder='./data/1_within_sample/train_data',
        model_path='./data/1_within_sample/train_out/E1000best_model.h5',
    """
    if isinstance(cell_embedding_ad, str) or isinstance(cell_embedding_ad, Path):
        cell_embedding_ad = Path(cell_embedding_ad)
        rna = sc.read_h5ad(cell_embedding_ad)
    elif isinstance(cell_embedding_ad, anndata.AnnData):
        rna = cell_embedding_ad
    else:
        raise ValueError(f"cell_embedding_ad must be a str, Path or anndata.AnnData!")
    input_folder = Path(input_folder)
    model_path = Path(model_path)
    output_path = Path(output_path)
    cellembed_raw = str(cellembed_raw)
    celltype = str(celltype)

    if not input_folder.exists():
        raise FileNotFoundError(f"Data folder {input_folder} not found! Run XChrom_preprocess.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model path {model_path} not found! Run XChrom_train.py first.")
    if not cell_embedding_ad.exists():
        raise FileNotFoundError(f"Cell embedding adata path {cell_embedding_ad} not found! Please check the path.")
    os.makedirs(output_path, exist_ok=True)
    if not celltype in rna.obs.columns:
        raise ValueError(f"Cell type {celltype} not found in the cell embedding adata!")
    if cellembed_raw not in rna.obsm:
        raise ValueError(f"Cell embeddings key {cellembed_raw} not found in the cell embeddings adata!")
    rna.obsm['zscore32_perpc'] = stats.zscore(np.array(rna.obsm[cellembed_raw]), axis=0) 
        # Actually, after model training, the rna local file will be updated, adding the cell_input_key, and the zscore does not need to be recalculated.
    ad = anndata.read_h5ad(f'{input_folder}/ad.h5ad')  ## peaks × cells
    assert ad.obs.index.equals(rna.obs.index), "Indexes differ!"
    ad.obsm['zscore32_perpc'] = rna.obsm['zscore32_perpc']
    ad.obs[celltype]=rna.obs[celltype]
    ad.obs['b_zscore'] = np.full(ad.shape[0],1)

    gen = Generator(
        seq_path = input_folder/'all_seqs.h5',
        adata = ad,
        cell_input_key = 'zscore32_perpc'
    )
    ds = gen.create_dataset(shuffle=False)
    model = XChrom_model(n_cells=ad.shape[0],show_summary=False)
    model.load_weights(model_path)
    pred = model.predict(ds)
    print('Denoise done! denoise shape is:',pred.shape)

    adp = ad.copy()
    adp.X = pred.transpose(1,0)

    ## XChrom impute 
    ad1 = adp.copy()
    ad1 = calc_pca(ad1)
    ns100,ls100= calc_nsls_score(rna,ad1,100,celltype,use_rep_rna = cellembed_raw,use_rep_atac='X_pca')
    print(f'neighbor score(100)={ns100:.4f},label score(100)={ls100:.4f}')

    ns50,ls50= calc_nsls_score(rna,ad1,50,celltype,use_rep_rna = cellembed_raw,use_rep_atac='X_pca')
    print(f'neighbor score(50)={ns50:.4f},label score(50)={ls50:.4f}')

    ns10,ls10= calc_nsls_score(rna,ad1,10,celltype,use_rep_rna = cellembed_raw,use_rep_atac='X_pca')
    print(f'neighbor score(10)={ns10:.4f},label score(10)={ls10:.4f}')

    if plot_umap:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        sc.pl.umap(ad1, color=celltype, ax=ax, show=False)
        plt.savefig(output_path/'denoise_umap.pdf', format='pdf', dpi=300)
    if save_pred:
        ad1.write_h5ad(output_path/'denoise_impute.h5ad')
    return {
        'ns100': ns100,
        'ls100': ls100,
        'ns50': ns50,
        'ls50': ls50,
        'ns10': ns10,
        'ls10': ls10
    }

# evaluate model performance in cross-samples scenario

def crosssamples_aucprc(
    cell_embedding_ad:Union[str, Path, anndata.AnnData],
    input_folder:Union[str, Path] = './test_data',
    output_path:Union[str, Path] = './eval_out',
    model_path:Union[str, Path] = './train_out/E1000best_model.h5',
    cellembed_raw:str = 'X_pca_harmony',
    save_pred:bool = False,
    scatter_plot:bool = False
    )->dict:
    """
    Evaluate the performance in cross-samples scenario, calculate auROC & auPRC for overall, per-cell and per-peak.
    
    Parameters
    ----------
    cell_embedding_ad: str or Path or anndata.AnnData
        Path to the cell embedding adata file. provide cell input embedding for XChrom model prediction.
    input_folder: str or Path
        Path to the test data folder. Should generate by XChrom_preprocess.py. 
        Must contain 'splits.h5', 'ad.h5ad', 'all_seqs.h5'.
    output_path: str or Path
        Path to the output folder.
    model_path: str or Path
        Path to the trained model.
    cellembed_raw: str
        Key of the raw cell input embedding in the cell embedding adata,to generate model input.
    save_pred: bool 
        Whether to save the prediction matrix with npy format.
    scatter_plot: bool
        Whether to plot the UMAP of the test cells.
    
    Returns
    -------
    metrics: dict
        Dictionary containing overall auROC, per-cell auROC, per-peak auROC, overall auPRC, per-cell auPRC, per-peak auPRC.
    
    Examples
    --------
    >>> import xchrom as xc
    metrics1 = xc.tl.crosssamples_aucprc(
        cell_embedding_ad='./data/2_cross_samples/test_rna_harmony.h5ad',
        input_folder='./data/2_cross_samples/test_data',
        model_path='./data/2_cross_samples/train_out/E1000best_model.h5',
        output_path='./data/2_cross_samples/eval_out',
        cellembed_raw='X_pca_harmony',
        save_pred=True,
        scatter_plot=True
        )
    """
    
    if isinstance(cell_embedding_ad, str) or isinstance(cell_embedding_ad, Path):
        cell_embedding_ad = Path(cell_embedding_ad)
        rna = sc.read_h5ad(cell_embedding_ad)
    elif isinstance(cell_embedding_ad, anndata.AnnData):
        rna = cell_embedding_ad
    else:
        raise ValueError(f"cell_embedding_ad must be a str, Path or anndata.AnnData!")
    input_folder = Path(input_folder)
    output_path = Path(output_path)
    model_path = Path(model_path)
    cellembed_raw = str(cellembed_raw)
    os.makedirs(output_path, exist_ok=True)
    if cellembed_raw not in rna.obsm:
        raise ValueError(f"Cell embedding key {cellembed_raw} not found in the cell embedding adata!")
    rna.obsm['zscore32_perpc'] = stats.zscore(np.array(rna.obsm[cellembed_raw]), axis=0) 

    ad = anndata.read_h5ad(f'{input_folder}/ad.h5ad')  ## peaks × cells
    assert ad.obs.index.equals(rna.obs.index), "Indexes differ!"
    ad.obsm['zscore32_perpc'] = rna.obsm['zscore32_perpc']
    m = sparse.load_npz(f'{input_folder}/m.npz').tocsr()
    gen = Generator(
        seq_path = input_folder/'all_seqs.h5',
        adata = ad,
        cell_input_key = 'zscore32_perpc',
        m = m,
    )
    ds = gen.create_dataset(shuffle=False)
    model = XChrom_model(n_cells=ad.shape[0],show_summary=False)
    model.load_weights(model_path)
    pred = model.predict(ds)
    print('Predict done! prediction shape is:',pred.shape)
    m_to1 = m.copy()
    m_to1[m_to1 != 0] = 1
    true_01matrix = m_to1.T.toarray()
    pred_matrix = pred.T

    ## calculate auc and pr
    ## -1 Calculate overall auROC & auPRC 
    overall_metrics = calc_auc_pr(true_01matrix, pred_matrix, 'overall')
    print(f"Overall auROC: {overall_metrics['auroc']:.4f}, auPRC: {overall_metrics['auprc']:.4f}")
    
    ## -2 Calculate per cell auRPC & auPRC
    percell_metrics = calc_auc_pr(true_01matrix, pred_matrix, 'percell')
    print(f"Per-cell auROC: {percell_metrics['auroc']:.4f}, auPRC: {percell_metrics['auprc']:.4f}")
    print(f"Valid cells: {percell_metrics['n_cells']}")

    ## -3 Calculate per peak auROC & auPRC
    perpeak_metrics = calc_auc_pr(true_01matrix, pred_matrix, 'perpeak')
    print(f"Per-peak auROC: {perpeak_metrics['auroc']:.4f}, auPRC: {perpeak_metrics['auprc']:.4f}")
    print(f"Valid peaks: {perpeak_metrics['n_peaks']}")

    if save_pred:
        np.save(output_path/'crosssamples_pred.npy', pred)
    if scatter_plot:
        plot_percell_aucprc(true_01matrix, pred_matrix, out_file=output_path/'crosssamples_percell_aucprc_scatterplot.pdf')
        plot_perpeak_aucprc(true_01matrix, pred_matrix, out_file=output_path/'crosssamples_perpeak_aucprc_scatterplot.pdf')
    return {
        'overall_auroc': overall_metrics['auroc'],
        'overall_auprc': overall_metrics['auprc'],
        'percell_auroc': percell_metrics['auroc'],
        'percell_auprc': percell_metrics['auprc'],
        'perpeak_auroc': perpeak_metrics['auroc'],
        'perpeak_auprc': perpeak_metrics['auprc']
    }
        
def crosssamples_nsls(
    cell_embedding_ad:Union[str, Path, anndata.AnnData],
    input_folder:Union[str, Path] = './test_data',
    output_path:Union[str, Path] = './eval_out',
    model_path:Union[str, Path] = './train_out/E1000best_model.h5',
    cellembed_raw:str = 'X_pca_harmony',
    celltype:str = 'cell_type',
    save_pred:bool = False,
    plot_umap:bool = False,
    **kwargs
    ):
    """
    Evaluate the performance in cross-samples scenario, calculate neighbor score(k=10,50,100) and label score(k=10,50,100) for all cells.
    
    Parameters
    ----------
    cell_embedding_ad: str or Path or anndata.AnnData
        Path to the cell embedding adata file. provide cell input embedding for XChrom model prediction.
    input_folder: str or Path
        Path to the test data folder. Should generate by XChrom_preprocess.py. 
        Must contain 'splits.h5', 'ad.h5ad', 'all_seqs.h5'.
    output_path: str or Path
        Path to the output folder.
    model_path: str or Path
        Path to the trained model.
    cellembed_raw: str
        Key of the raw cell input embedding in the cell embedding adata,to calculate RNA neighbors and generate model input.
    celltype: str
        Key of the cell type label in the cell embedding adata.
    save_pred: bool
        Whether to save the prediction matrix with npy format.
    plot_umap: bool
        Whether to plot the UMAP of the test cells.

    Returns
    -------
    metrics: dict
        Dictionary containing neighbor score(k=10,50,100) and label score(k=10,50,100) for all cells.
    
    Examples
    --------
    >>> import xchrom as xc
    metrics2 = xc.tl.crosssamples_nsls(
        cell_embedding_ad='./data/2_cross_samples/test_rna_harmony.h5ad',
        input_folder='./data/2_cross_samples/test_data',
        model_path='./data/2_cross_samples/train_out/E1000best_model.h5',
        output_path='./data/2_cross_samples/eval_out',
        cellembed_raw='X_pca_harmony',
        use_rep_rna='X_pca',
        celltype = 'cell_type',
        save_pred=True,
        plot_umap=True
        )
    """
    
    if isinstance(cell_embedding_ad, str) or isinstance(cell_embedding_ad, Path):
        cell_embedding_ad = Path(cell_embedding_ad)
        rna = sc.read_h5ad(cell_embedding_ad)
    elif isinstance(cell_embedding_ad, anndata.AnnData):
        rna = cell_embedding_ad
    else:
        raise ValueError(f"cell_embedding_ad must be a str, Path or anndata.AnnData!")
    input_folder = Path(input_folder)
    model_path = Path(model_path)
    output_path = Path(output_path)
    cellembed_raw = str(cellembed_raw)
    celltype = str(celltype)
    os.makedirs(output_path, exist_ok=True)
    use_rep_rna = kwargs.pop('use_rep_rna', cellembed_raw)

    rna.obsm['zscore32_perpc'] = stats.zscore(np.array(rna.obsm[cellembed_raw]), axis=0) 
    ad = anndata.read_h5ad(f'{input_folder}/ad.h5ad')  ## peaks × cells
    assert ad.obs.index.equals(rna.obs.index), "Indexes differ!"
    ad.obsm['zscore32_perpc'] = rna.obsm['zscore32_perpc']
    ad.obs[celltype]=rna.obs[celltype]
    ad.obs['b_zscore'] = np.full(ad.shape[0],1)
    m = sparse.load_npz(f'{input_folder}/m.npz').tocsr()

    gen = Generator(
        seq_path = input_folder/'all_seqs.h5',
        adata = ad,
        cell_input_key = 'zscore32_perpc',
        m = m,
    )
    ds = gen.create_dataset(shuffle=False)
    model = XChrom_model(n_cells=ad.shape[0],show_summary=False)
    model.load_weights(model_path)
    pred = model.predict(ds)
    print('Predict done! prediction shape is:',pred.shape)
    adp = ad.copy()
    adp.X = pred.transpose(1,0)

    ad1 = adp.copy()
    ad1 = calc_pca(ad1)
    ns100,ls100= calc_nsls_score(rna,ad1,100,celltype,use_rep_rna = use_rep_rna,use_rep_atac='X_pca',**kwargs)
    print(f'neighbor score(100)={ns100:.4f},label score(100)={ls100:.4f}')

    ns50,ls50= calc_nsls_score(rna,ad1,50,celltype,use_rep_rna = use_rep_rna,use_rep_atac='X_pca',**kwargs)
    print(f'neighbor score(50)={ns50:.4f},label score(50)={ls50:.4f}')

    ns10,ls10= calc_nsls_score(rna,ad1,10,celltype,use_rep_rna = use_rep_rna,use_rep_atac='X_pca',**kwargs)
    print(f'neighbor score(10)={ns10:.4f},label score(10)={ls10:.4f}')
    if plot_umap:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        sc.pl.umap(ad1, color=celltype, ax=ax, show=False)
        plt.savefig(output_path/'crosssamples_umap.pdf', format='pdf', dpi=300)
    if save_pred:
        ad1.write_h5ad(output_path/'crosssamples_impute.h5ad')
    return {
        'ns100': ns100,
        'ls100': ls100,
        'ns50': ns50,
        'ls50': ls50,
        'ns10': ns10,
        'ls10': ls10
    }
