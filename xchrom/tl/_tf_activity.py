import os
from pathlib import Path
from typing import Union
import anndata
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from scipy import stats
from Bio import SeqIO
from .._utils import setup_seed
from ..tr._utils import XChrom_model, Generator
from ._utils import fasta_to_h5, generate_h5_files, generate_tf_activity_data
import scipy.sparse

def calc_tf_activity(
    motif_dir: Union[str, Path],
    background_fasta: Union[str, Path],
    model_path: Union[str, Path],
    ad_rna: anndata.AnnData,
    output_file: Union[str, Path] = Path('/tf_activity.h5ad'),
    cell_embed_raw: str = 'X_pca_harmony',
    regenerate_motif_h5: bool = False,
    regenerate_bg_h5: bool = False,
    seq_len: int = 1344,
    seed: int = 20,
    **model_kwargs
    ) -> anndata.AnnData:
    """
    Calculate motif insertion scores.
    
    Parameters
    ----------
    motif_dir: Union[str, Path]
        The path to the directory containing motif insertion fasta files
    background_fasta: Union[str, Path] 
        The path to the background sequence fasta file
    model_path: Union[str, Path]
        The path to the trained XChrom model weights file
    ad_rna: anndata.AnnData
        scRNA-seq data, must contain raw cell embedding in ad_rna.obsm
    output_file: Union[str, Path], default='./tf_activity.h5ad'
        Output file path for TF activity results
    cell_embed_raw: str, default='X_pca_harmony'
        The key name of raw cell embedding from ad_rna.obsm
    regenerate_motif_h5: bool, default False
        Whether to regenerate motif insertion sequence h5 files, if False, will use existing h5 files
    regenerate_bg_h5: bool, default False
        Whether to regenerate background sequence h5 files, if False, will use existing h5 file
    seq_len: int, default 1344
        Sequence length for background sequence and motif insertion sequence
    seed: int, default 20
        Random seed
    **model_kwargs: dict
        Additional parameters passed to XChrom_model
        
    Returns
    -------
    anndata.AnnData
        TF activity results, X is the activity matrix of cells × motifs
        save to './tf_activity.h5ad' by default
    
    Examples
    --------
    >>> import xchrom as xc
    >>> tf_act = xc.tl.calc_tf_activity(
    ...     motif_dir='./motif_fasta/',
    ...     background_fasta='./shuffled_peaks.fasta',
    ...     model_path='./best_model.h5', 
    ...     ad_rna=m1d1_rna,
    ...     cell_embed_raw='X_pca_harmony', 
    ...     regenerate_bg_h5=True,
    ...     regenerate_motif_h5=True,
    ...     seed=20
    ... )
    """
    setup_seed(seed=seed)
    print("=== Start calculating TF Activity ===")
    # 1. Prepare raw cell embedding
    print("1. Prepare raw cell embedding...")
    if cell_embed_raw not in ad_rna.obsm:
        raise ValueError(f"The raw cell embedding '{cell_embed_raw}' is not found in ad_rna.obsm")
    
    cell_embed = stats.zscore(np.array(ad_rna.obsm[cell_embed_raw]), axis=0)
    
    # 2. Generate motif insertion sequence h5 files
    print("2. Generate motif insertion sequence h5 files...")
    if regenerate_motif_h5:
        h5_files, motif_names = generate_h5_files(motif_dir)
    else:
        h5_files = []
        motif_names = []
        files = sorted([f for f in os.listdir(motif_dir) if f.endswith('.h5')])
        if not files:
            print("No h5 files found, generating...")
            h5_files, motif_names = generate_h5_files(motif_dir)
        else:
            for h5 in files:
                h5_path = os.path.join(motif_dir, h5)
                h5_files.append(h5_path)
                motif_names.append(os.path.splitext(h5)[0])
    
    print(f"Found {len(h5_files)} motif files")
    
    # 3. Generate background sequence h5 file
    print("3. Generate background sequence h5 file...")
    bg_h5_path = os.path.splitext(background_fasta)[0] + ".h5"
    if regenerate_bg_h5 or not os.path.exists(bg_h5_path):
        print("Generating background sequence h5 file...")
        fasta_to_h5(background_fasta, seq_len=seq_len)
    num_seqs = len(list(SeqIO.parse(background_fasta, "fasta")))
    
    # 4. Create virtual data for dataset creation
    print("4. Prepare data for XChrom model input...")
    obs_info = ad_rna.obs
    # Create virtual data for dataset generation
    zero_matrix = np.zeros((len(obs_info), num_seqs))
    data_virtual = anndata.AnnData(X=zero_matrix, obs=obs_info)
    data_virtual.obsm['zscore32_perpc'] = cell_embed
    data_virtual.obs['b_zscore'] =  np.full(data_virtual.shape[0], 1)
    
    # Pre-compute sparse matrix to avoid repeated warnings
    print("Converting adata.X to dense array. For large datasets, consider pre-computing and saving as sparse matrix.")
    m_sparse = scipy.sparse.csr_matrix(data_virtual.X.T)
    
    # 5. Load trained model
    print("5. Load trained model...")
    n_cells = data_virtual.shape[0]
    model = XChrom_model(n_cells=n_cells, show_summary=False, **model_kwargs)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    model.load_weights(model_path)
    
    # Create prediction model (remove the last layer)
    prediction_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    # 6. Calculate background sequence prediction
    print("6. Calculate background sequence prediction...")

    gen = Generator(
        seq_path=bg_h5_path,
        adata=data_virtual,
        cell_input_key='zscore32_perpc',
        peakid=None,
        m=m_sparse,
        batch_size=128
    )
    bg_ds = gen.create_dataset()
    bg_pred = prediction_model.predict(bg_ds)
    bg_pred_mean = bg_pred.mean(axis=0)
    
    # 7. Calculate motif insertion scores
    print("7. Calculate motif insertion scores...")
    h5_files, motif_names = zip(*sorted(zip(h5_files, motif_names)))
    
    n_motifs = len(h5_files)
    tf_scores_raw = np.zeros((n_cells, n_motifs))
    print(f"Processing {n_motifs} motif files...")
    for i, motif_path in enumerate(tqdm(h5_files)):
        try:
            # Create tf.data.Dataset for each motif file
            gen = Generator(
                seq_path=motif_path,
                adata=data_virtual,
                cell_input_key='zscore32_perpc',
                peakid=None,
                m=m_sparse,
                batch_size=128)
            motif_ds = gen.create_dataset()
            # Predict motif insertion effect
            motif_pred = prediction_model.predict(motif_ds)
            motif_mean = motif_pred.mean(axis=0)
            # Calculate the difference between motif insertion and background
            delta = motif_mean - bg_pred_mean
            tf_scores_raw[:, i] = delta
        except Exception as e:
            print(f"Error processing {motif_path}: {str(e)}")
            tf_scores_raw[:, i] = 0
    
    # 8. Create result AnnData object
    print("8. Save results...")
    tf_act = anndata.AnnData(
        X=tf_scores_raw,  # cells × motifs
        obs=ad_rna.obs.copy(),
        var=pd.DataFrame(index=list(motif_names))
    )
    
    tf_act.var["motif_name"] = tf_act.var.index
    
    # Save results
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tf_act.write_h5ad(output_file)
    
    print(f"=== Done! Results saved to: {output_file} ===")
    print(f"TF activity matrix shape: {tf_act.shape} (cells × motifs)")
    
    return tf_act
