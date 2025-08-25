#!/usr/bin/env python
"""
XChrom model training script

Run as a standalone script:
    python XChrom_train.py --input_folder ./data/1_within_sample/train_data/ 
    --cell_embedding_ad ./data/1_within_sample/m_brain_paired_rna.h5ad 
    --cellembed_raw 'X_pca' 
    --out_path ./data/1_within_sample/train_out/
    --trackscore
    --celltype 'pc32_leiden'
    --epochs 1000
    --save_freq 1000
    --verbose 0  # silent mode, no progress bar

Import as a module:
    >>> import xchrom as xc
    history = xc.tr.train_XChrom(
        input_folder='./data/1_within_sample/train_data/',
        cell_embedding_ad='./data/1_within_sample/m_brain_paired_rna.h5ad',
        cellembed_raw='X_pca',
        out_path='./data/1_within_sample/train_out/',
        trackscore = True,
        celltype = 'pc32_leiden',
        epochs = 1000,
        save_freq = 1000,
        verbose = 0  # silent mode, no progress bar
        )
"""

import anndata
import h5py
import tensorflow as tf
import numpy as np
import scipy.sparse as sparse
import pickle
import configargparse
import random
from pathlib import Path
from typing import Union, Dict, Any, Literal
from scipy import stats
import os
from .._utils import setup_seed
from ._utils import Generator, XChrom_model, Callback_TrackScore, Callback_SaveModel

# os.environ["CUDA_VISIBLE_DEVICES"]="3,2,1,0"  

def train_XChrom(
    input_folder: Union[str, Path],
    cell_embedding_ad: Union[str, Path], 
    out_path: Union[str, Path] = './train_out',
    bottleneck: int = 32,
    batch_size: int = 128,
    lr: float = 0.01,
    epochs: int = 1000,
    save_freq: int = 1000,
    trackscore: bool = False,
    celltype: str = 'celltype',
    seed: int = 20,
    train_split: float = 0.9,
    cellembed_raw: str = 'X_pca',
    verbose: Literal[0, 1, 2] = 1,
    print_scores: bool = False,
    **kwargs
    ) -> Dict[str, Any]:
    """
    Train XChrom model
    
    Parameters
    ----------
    input_folder: Union[str, Path]
        Preprocessed data folder, should contain: trainval_seqs.h5, splits.h5, ad_trainval.h5ad, m_trainval.npz
    cell_embedding_ad: Union[str, Path]
        scRNA-seq data file path containing raw cell embedding
    out_path: Union[str, Path], default 'train_out'
        Output path
    bottleneck: int, default 32
        Bottleneck layer size,should be the same as the dimension of raw cell embedding
    batch_size: int, default 128
        Batch size
    lr: float, default 0.01
        Learning rate
    epochs: int, default 1000
        Number of training epochs
    save_freq: int, default 1000
        Model saving frequency
    trackscore: bool, default False
        Whether to compute score metrics every epoch
    celltype: str, default 'cell_type'
        Cell type label column name (used when trackscore=True)
    seed: int, default 20
        Random seed
    train_split: float, default 0.9
        Training set/validation set ratio
    cellembed_raw: str, default 'X_pca'
        Raw cell embedding key in cell embedding adata
    verbose: int, default 1
        Training verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch
    print_scores: bool, default False
        Whether to print ns,ls scores every epoch when trackscore=True
    **kwargs: dict
        Additional parameters
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing training history and model information
        
    Examples
    --------
    >>> import xchrom as xc
    >>> history = xc.tr.train_XChrom(
        input_folder='./data/1_within_sample/train_data/',
        cell_embedding_ad='./data/1_within_sample/m_brain_paired_rna.h5ad',
        cellembed_raw='X_pca',
        out_path='./data/1_within_sample/train_out/',
        trackscore = True,
        celltype = 'pc32_leiden',
        epochs = 1000,
        save_freq = 1000,
        verbose = 0,  # silent mode, no progress bar
        print_scores = False  # whether to print ns,ls scores every epoch when trackscore=True
        )
    """
    
    setup_seed(seed)
    
    # Convert paths to Path objects
    input_folder = Path(input_folder)
    cell_embedding_ad = Path(cell_embedding_ad)
    out_path = Path(out_path)
    
    # Verify input files
    required_files = [
        'trainval_seqs.h5', 'splits.h5', 'ad_trainval.h5ad', 'm_trainval.npz'
    ]
    for file in required_files:
        if not (input_folder / file).exists():
            raise FileNotFoundError(f"Required file not found: {input_folder / file}")
    
    if not cell_embedding_ad.exists():
        raise FileNotFoundError(f"Cell embedding file not found: {cell_embedding_ad}")
    
    # Create output directory
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path / 'epoch_model', exist_ok=True)
    print("=== Start training XChrom model ===")
    print(f"Input folder: {os.path.abspath(input_folder)}")
    print(f"Cell embedding file: {os.path.abspath(cell_embedding_ad)}")
    print(f"Raw cell embedding key: {cellembed_raw}")
    print(f"Output path: {os.path.abspath(out_path)}")
    print(f"Model parameters: bottleneck={bottleneck}, batch_size={batch_size}, lr={lr}")
    
    # 1. Load raw cell embedding and make z-score normalization
    print("1. Load raw cell embedding and make z-score normalization...")
    rna_ad = anndata.read_h5ad(cell_embedding_ad)
    if cellembed_raw not in rna_ad.obsm:
        raise ValueError(f"Embedding key '{cellembed_raw}' not found in RNA data")
    print(f"Raw cell embedding saved to: {os.path.abspath(cell_embedding_ad)}.obsm['{cellembed_raw}']")
    zscore32_perpc = stats.zscore(np.array(rna_ad.obsm[cellembed_raw]), axis=0)
    rna_ad.obsm['zscore32_perpc'] = zscore32_perpc
    rna_ad.write_h5ad(cell_embedding_ad)
    print(f"Initial cell embedding saved to: {os.path.abspath(cell_embedding_ad)}.obsm['zscore32_perpc']")
    print(f"Initial cell embedding shape: {rna_ad.obsm['zscore32_perpc'].shape}")
    
    # 2. Load training data
    print("2. Load training data...")
    with h5py.File(input_folder / 'splits.h5', 'r') as hf:
        trainval_cellid = hf['trainval_cell'][:]
        rna_trainval = rna_ad[trainval_cellid, :]
    trainval_seq = str(input_folder / 'trainval_seqs.h5')
    ad_trainval = anndata.read_h5ad(input_folder / 'ad_trainval.h5ad')
    m_trainval = sparse.load_npz(input_folder / 'm_trainval.npz').tocsr()
    
    # Verify data consistency
    if not ad_trainval.obs.index.equals(rna_trainval.obs.index):
        raise ValueError("scATAC and scRNA data cell indices do not match")
    ad_trainval.obsm['zscore32_perpc'] = rna_trainval.obsm['zscore32_perpc']
    
    # 3. Prepare training/validation split
    print("3. Prepare train/val data split...")
    peak_ids = list(range(m_trainval.shape[0]))
    random.shuffle(peak_ids)
    train_size = int(train_split * len(peak_ids))
    train_id = sorted(peak_ids[:train_size])
    val_id = sorted(peak_ids[train_size:])
    print(f"Training peak number: {len(train_id)}, Validation peak number: {len(val_id)}")
    
    # Prepare data subset
    ad_train = ad_trainval[:, train_id]
    ad_val = ad_trainval[:, val_id]
    m_train = m_trainval[train_id, :]
    m_val = m_trainval[val_id, :]
    train_cell = ad_train.shape[0]
    
    # 4. Create TensorFlow dataset
    print("4. Create TensorFlow dataset...")
    gen1 = Generator(
        adata=ad_train,
        seq_path=trainval_seq,
        cell_input_key='zscore32_perpc',
        peakid=train_id,
        m=m_train,
        batch_size=batch_size)
    train_ds = gen1.create_dataset(shuffle=True)
    gen2 = Generator(
        adata=ad_val,
        seq_path=trainval_seq,
        cell_input_key='zscore32_perpc',
        peakid=val_id,
        m=m_val,
        batch_size=batch_size)
    val_ds = gen2.create_dataset(shuffle=False)
    
    # 5. Build and compile model
    print("5. Build and compile model...")
    model = XChrom_model(n_cells=train_cell, cell_vec=bottleneck, **kwargs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=10000, decay_rate=0.9
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=0.95, beta_2=0.9995
        ),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy', 
            tf.keras.metrics.AUC(name='auc', curve='ROC', multi_label=False), 
            tf.keras.metrics.AUC(name='pr', curve='PR', multi_label=False)
        ]
    )
    
    # 6. Set training callbacks
    print("6. Set training callbacks...")
    best_model_path = out_path / 'E1000best_model.h5'
    save_freq = int(save_freq)
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_model_path), save_best_only=True, save_weights_only=True, 
            monitor='auc', mode='max',restore_best_weights=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='auc', min_delta=1e-6, mode='max', patience=50, verbose=1
        ),
        Callback_SaveModel(
            str(out_path / 'epoch_model' / 'epoch{epoch:04d}_model.h5'), save_freq
        )
    ]
    
    if trackscore:
        if celltype not in rna_trainval.obs.columns:
            raise ValueError(f"Cell type column '{celltype}' not found in RNA data, which is required when trackscore=True")
        ad_trainval.obs[celltype] = rna_trainval.obs[celltype]
        callbacks_list.append(
            Callback_TrackScore(rna_trainval, ad_trainval, model, print_scores=print_scores, use_rep_rna=cellembed_raw, label=celltype,cell_input_key='zscore32_perpc')
        )
    
    # 7. Start training
    print("7. Start training...")
    print(f"Model will be saved to: {best_model_path}")
    
    history = model.fit(
        train_ds, 
        batch_size=batch_size,
        epochs=epochs, 
        validation_data=val_ds, 
        callbacks=callbacks_list,
        use_multiprocessing=True,
        workers=4,
        verbose=verbose
    )
    
    # 8. Save results
    print("8. Save training results...")
    history_path = out_path / 'history.pickle'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"=== Training completed! ===")
    print(f"Best model: {os.path.abspath(best_model_path)}")
    print(f"Training history: {os.path.abspath(history_path)}")
    return {
        'history': history.history,
        'model': model,
        'best_model_path': os.path.abspath(best_model_path),
        'train_cells_number': train_cell,
        'train_peaks_number': len(train_id),
        'val_peaks_number': len(val_id)
    }


def make_parser():
    """Create command line argument parser"""
    parser = configargparse.ArgParser(
        description="Train XChrom model - can be run as a standalone script or imported as a module"
    )
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Preprocessed data folder, should contain: trainval_seqs.h5, splits.h5, ad_trainval.h5ad, m_trainval.npz')
    parser.add_argument('--cell_embedding_ad', type=str, required=True,
                       help='scRNA-seq data file path containing raw cell embedding')
    parser.add_argument('--out_path', type=str, default='train_out',
                       help='Output path, default to ./train_out/')
    parser.add_argument('--bottleneck', type=int, default=32,
                       help='Bottleneck layer size, default to 32')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training, default to 128')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate, default to 0.01')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs, default to 1000')
    parser.add_argument('--save_freq', type=int, default=1000,
                       help='Model saving frequency, default to 1000 epochs, just save the best model')
    parser.add_argument('--trackscore', action='store_true',
                       help='Whether to compute ns,ls score metrics every epoch, default to False')
    parser.add_argument('--celltype', type=str, default='cell_type',
                       help='Cell type label column name, required when trackscore=True')
    parser.add_argument('--seed', type=int, default=20,
                       help='Random seed, default to 20')
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Training set/validation set ratio, default to 0.9')
    parser.add_argument('--cellembed_raw', type=str, default='X_pca',
                       help='Raw cell embedding key in RNA data, default to X_pca')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                       help='Training verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch, default to 1')
    return parser


def main():
    """Command line entry function"""
    parser = make_parser()
    args = parser.parse_args()
    
    # call training function
    try:
        result = train_XChrom(
            input_folder=args.input_folder,
            cell_embedding_ad=args.cell_embedding_ad,
            out_path=args.out_path,
            bottleneck=args.bottleneck,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            save_freq=args.save_freq,
            trackscore=args.trackscore,
            celltype=args.celltype,
            seed=args.seed,
            train_split=args.train_split,
            cellembed_raw=args.cellembed_raw,
            verbose=args.verbose
        )
        
        print(f"\n=== Training statistics ===")
        print(f"Training cell number: {result['train_cells_number']}")
        print(f"Training peak number: {result['train_peaks_number']}")
        print(f"Validation peak number: {result['val_peaks_number']}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()