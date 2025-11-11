# plot functions
import numpy as np
from pathlib import Path
from typing import Union
import os
import math
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
if not hasattr(mpl.colormaps, "get_cmap") and hasattr(mpl.cm, "get_cmap"):
    mpl.colormaps.get_cmap = mpl.cm.get_cmap
import anndata
import scanpy as sc
from .._utils import setup_seed

def plot_train_history(
    history:dict,
    savefig:bool = False,
    out_file:Union[str, Path] = './train_out/train_history_plot.pdf'
    ):
    out_file = Path(out_file)
    os.makedirs(out_file.parent, exist_ok=True) 
    if out_file.suffix != '.pdf':
        out_file = out_file.with_suffix('.pdf')
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_auc = history['auc']
    train_pr = history['pr']
    val_auc = history['val_auc']
    val_pr = history['val_pr']

    if history.get('neighbor_score') is not None and history.get('label_score') is not None:
        neighbor_score = history['neighbor_score']
        label_score = history['label_score']

        plt.figure(figsize=(15, 6))
        plt.subplot(1,3,1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1,3,2)
        plt.plot(train_auc, label='Training AUC')
        plt.plot(val_auc, label='Validation AUC')
        plt.plot(train_pr, label='Training PR')
        plt.plot(val_pr, label='Validation PR')
        plt.title('Training/Validation PR and AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC/PR')
        plt.legend()
        plt.subplot(1,3,3)
        plt.plot(neighbor_score, label='Neighbor Score')
        plt.plot(label_score, label='Label Score')
        plt.title('Neighbor and Label Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        if savefig:
            plt.savefig(out_file, format='pdf', dpi=300)
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(train_auc, label='Training AUC')
        plt.plot(val_auc, label='Validation AUC')
        plt.plot(train_pr, label='Training PR')
        plt.plot(val_pr, label='Validation PR')
        plt.title('Training/Validation PR and AUC')
        plt.legend()
        if savefig:
            plt.savefig(out_file, format='pdf', dpi=300)
        plt.show()
        

def plot_logo(
    m:np.ndarray, 
    ymin:float, 
    ymax:float, 
    ax:plt.Axes, 
    title:str = None
    ):
    """
    Plot the relative importance of the given matrix per position.

    Parameters
    ----------
    m: np.ndarray
        The matrix to plot the logo, shape (n_cells, seq_len, 4).
    ymin: float
        The minimum value of the y-axis.
    ymax: float
        The maximum value of the y-axis.
    ax: plt.Axes
        The axes to plot the logo.
    title: str
        The title of the plot.

    """
    import logomaker
    
    nn_logo = logomaker.Logo(m, ax=ax, baseline_width=0)  
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=['left'], visible=True, bounds=[ymin, ymax])
    ax.set_ylabel('saliency', labelpad=-1)
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)

def plot_perpeak_aucprc(
    true_matrix:np.ndarray,
    pred_matrix:np.ndarray,
    out_file:Union[str, Path] = './eval_out/perpeak_aucprc_scatterplot.pdf'
    ):
    """
    Plot the per-peak auROC and auPRC scatter plot.
    Parameters
    ----------
    true_matrix: np.ndarray
        The true label matrix, shape (n_cells, n_peaks).
    pred_matrix: np.ndarray
        The prediction matrix, shape (n_cells, n_peaks).
    out_dir: Union[str, Path]
        The directory to save the plot.
    out_name: Union[str, Path]
        The name of the plot.

    Returns
    -------
    None
    """
    out_file = Path(out_file)
    os.makedirs(out_file.parent, exist_ok=True) 
    if out_file.suffix != '.pdf':
        out_file = out_file.with_suffix('.pdf')
    setup_seed(seed=20)  
    auc_values = []
    pr_values = []
    cell_cnts = []
    for j in range(pred_matrix.shape[1]):  # per peak
        y_true = true_matrix[:, j]
        y_pred = pred_matrix[:, j]
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 1:
            continue
        auc = roc_auc_score(y_true, y_pred)
        pr = average_precision_score(y_true, y_pred)
        cell_cnt = np.sum(y_true)
        auc_values.append(auc)
        pr_values.append(pr)
        cell_cnts.append(cell_cnt)
    avg_auc = sum(auc_values) / len(auc_values)
    avg_pr = sum(pr_values) / len(pr_values)
    log_cell_cnts = [math.log2(cnt) for cnt in cell_cnts if cnt > 0]  

    f = 20
    R_auc_matrix = np.corrcoef(auc_values, log_cell_cnts)
    R_pr_matrix = np.corrcoef(pr_values, log_cell_cnts)
    R_auc = R_auc_matrix[0, 1]
    R_pr = R_pr_matrix[0, 1]

    plt.figure(figsize=(10,7))
    plt.subplot(1, 2, 1)
    plt.scatter(auc_values, log_cell_cnts, color='blue', label='auROC',s=20, alpha=0.5)
    plt.xlabel('per peak auROC', fontsize=f)
    plt.ylabel('log2cell_count', fontsize=f)
    plt.title(f'mean = {avg_auc:.4f}', fontsize=f)
    plt.text(min(auc_values),max(log_cell_cnts)-0.5, f'R: {R_auc:.2f}', fontsize=f, color='blue')
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=f-4)

    plt.subplot(1, 2, 2)
    plt.scatter(pr_values, log_cell_cnts, color='red', label='auPR',s=20, alpha=0.5)
    plt.xlabel('per peak auPRC', fontsize=f)
    plt.ylabel('log2cell_count', fontsize=f)
    plt.title(f'mean = {avg_pr:.4f}', fontsize=f)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=f-4)
    plt.text(min(pr_values),max(log_cell_cnts)-0.5, f'R: {R_pr:.2f}', fontsize=f, color='red')
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95, top=0.9, bottom=0.1)
    plt.savefig(out_file, format='pdf', dpi=300)
    plt.show()
    
    

def plot_percell_aucprc(
    true_matrix:np.ndarray,
    pred_matrix:np.ndarray,
    out_file:Union[str, Path] = './eval_out/percell_aucprc_scatterplot.pdf'
    ):
    """
    Plot the per-cell auROC and auPRC scatter plot.
    Parameters  
    ----------
    true_matrix: np.ndarray
        The true label matrix, shape (n_cells, n_peaks).
    pred_matrix: np.ndarray
        The prediction matrix, shape (n_cells, n_peaks).
    out_file: Union[str, Path]
        The file to save the plot.

    Returns
    -------
    None
    """
    out_file = Path(out_file)
    os.makedirs(out_file.parent, exist_ok=True) 
    if out_file.suffix != '.pdf':
        out_file = out_file.with_suffix('.pdf')
    auc_values = []
    pr_values = []
    peak_cnts = []
    for k in range(pred_matrix.shape[0]):  # per cell
        y_true = np.array(true_matrix[k, :])
        y_pred = pred_matrix[k, :]
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 1:
            continue
        auc = roc_auc_score(y_true, y_pred)
        pr = average_precision_score(y_true, y_pred)
        peak_cnt = np.sum(y_true)
        auc_values.append(auc)
        pr_values.append(pr)
        peak_cnts.append(peak_cnt)
    avg_auc = sum(auc_values) / len(auc_values)
    avg_pr = sum(pr_values) / len(pr_values)
    log_peak_cnts = [math.log2(cnt) for cnt in peak_cnts if cnt > 0]

    f = 20
    R_auc_matrix = np.corrcoef(auc_values, log_peak_cnts)
    R_pr_matrix = np.corrcoef(pr_values, log_peak_cnts)
    R_auc = R_auc_matrix[0, 1]
    R_pr = R_pr_matrix[0, 1]

    plt.figure(figsize=(10,7))
    plt.subplot(1, 2, 1)
    plt.scatter(auc_values, log_peak_cnts, color='blue', label='auROC',s=20, alpha=0.5)
    plt.xlabel('per cell auROC', fontsize=f)
    plt.ylabel('log2peak_count', fontsize=f)
    plt.title(f'mean = {avg_auc:.4f}', fontsize=f)
    plt.text(min(auc_values),max(log_peak_cnts)-0.5, f'R: {R_auc:.2f}', fontsize=f, color='blue') 
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=f-4)

    plt.subplot(1, 2, 2)
    plt.scatter(pr_values, log_peak_cnts, color='red', label='auPR',s=20, alpha=0.5)
    plt.xlabel('per cell auPRC', fontsize=f)
    plt.ylabel('log2peak_count', fontsize=f)
    plt.title(f'mean = {avg_pr:.4f}', fontsize=f)
    plt.legend(fontsize=15)
    plt.text(min(pr_values),max(log_peak_cnts)-1, f'R: {R_pr:.2f}', fontsize=f, color='red')
    plt.tick_params(axis='both', labelsize=f-4)
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95, top=0.9, bottom=0.1)
    plt.savefig(out_file, format='pdf', dpi=300)
    plt.show()
    
    

def plot_motif_activity(
    cell_embedding_ad: anndata.AnnData,
    celltype_key: str,
    tf_act_raw: anndata.AnnData,
    motif_name: str,
    save_path: str = None
    ):
    """
    Plot the activity of a single motif and celltype on UMAP.
    
    Parameters
    ----------
    cell_embedding_ad: anndata.AnnData
        Anndata object containing reduced dimension embedding
    celltype_key: str
        Key for cell type in cell_embedding_ad.obs
    tf_act_raw: anndata.AnnData
        Anndata object containing raw TF activity data
    motif_name: str
        Name of the motif to plot
    save_path: str, optional
        Path to save the plot

    Returns
    -------
    None
    
    Examples
    --------
    >>> import xchrom as xc
    >>> xc.pl.plot_motif_activity(
        cell_embedding_ad = covid19_rna, 
        celltype_key = 'celltypeL0', 
        tf_act_raw = tf_act, 
        motif_name = 'RUNX3', 
        save_path = './RUNX3_activity.pdf'
        )
    """
    sc.set_figure_params(frameon=False)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['figure.facecolor'] = 'white'
    
    sc.pp.scale(tf_act_raw)
    ad = tf_act_raw[:, tf_act_raw.var['motif_name'] == motif_name]
    if ad.shape[1] == 0:
        raise ValueError(f"Motif '{motif_name}' not found in {tf_act_raw}")
    cell_embedding_ad.obs[f'{motif_name}_activity'] = ad.X.flatten()

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax0 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(cell_embedding_ad, color=celltype_key, ax=ax0, show=False)
    ax0.set_title(f'{celltype_key}')
    ax1 = fig.add_subplot(gs[0, 1])
    sc.pl.umap(cell_embedding_ad, color=f'{motif_name}_activity', ax=ax1, 
               cmap='coolwarm', vmin=-2, vmax=2, show=False)
    ax1.set_title(f'{motif_name}_activity')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    plt.show()
