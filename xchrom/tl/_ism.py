import numpy as np
import tensorflow as tf
import anndata
import scipy.stats as stats
from pathlib import Path
from typing import Union
from .._utils import make_bed_seqs, dna_1hot

# compute ism from sequence
def calc_ism(
    cell_embedding_ad:anndata.AnnData,
    XChrom_model:tf.keras.Model,
    seq_ref_1hot: np.ndarray,
    cellembed_raw:str = 'X_pca_harmony',
    layer_offset:int = -8
    ) -> np.ndarray:
    """
    Calculate ISM (In Silico Mutagenesis) for a single sequence
    
    Parameters
    ----------
    cell_embedding_ad: anndata.AnnData
        anndata object with Initial cell embeddings
    cellembed_raw: str
        Key of the raw cell input embedding in the cell embedding adata,to generate model input.
    seq_ref_1hot: np.ndarray
        One-hot encoded reference sequence, shape (seq_len, 4)
    XChrom_model: tf.keras.Model
        Trained XChrom model for sequence encoding
    layer_offset : int, optional (default is -8)
        Layer offset for extracting sequence embeddings from XChrom_model.
        Default is -8 (8th layer from the end). Adjust if model architecture changes.
        
    Returns
    -------
    np.ndarray
        ISM matrix showing nucleotide importance, shape (n_cells, seq_len, 4)

    Examples
    --------
    >>> seq_ref_1hot = dna_1hot("ATCGATCG" * 168)  # shape: (1344, 4)
    >>> XChrom_model = XChrom_model(n_cells=1, cell_vec=32)
    >>> XChrom_model.load_weights('model_weights.h5')
    >>> ism = calc_ism(seq_ref_1hot, XChrom_model,layer_offset=-8)
    >>> print(f"The ISM matrix is saved in ism")
    
    
    
    """
    cell_input = XChrom_model.input[1]
    x2 = XChrom_model.get_layer('final_cellembed').output 
    new_model1 = tf.keras.Model(inputs = cell_input,outputs = x2)
    if cellembed_raw not in cell_embedding_ad.obsm:
        raise ValueError(f"Cell embedding key {cellembed_raw} not found in the cell embedding adata!")
    cell_embedding_ad.obsm['zscore32_perpc'] = stats.zscore(np.array(cell_embedding_ad.obsm[cellembed_raw]), axis=0) 
    w = new_model1.predict(np.expand_dims(cell_embedding_ad.obsm['zscore32_perpc'], axis=0))
    w = np.array(tf.squeeze(w))

    layer_name=[]
    for layer in XChrom_model.layers:
        layer_name.append(layer.name)
    seq = XChrom_model.input[0]
    peak_embed = XChrom_model.get_layer(layer_name[layer_offset]).output
    new_model2 = tf.keras.Model(inputs = seq,outputs = peak_embed)

    # output ISM matrix, shape (n_cells, seq_len, 4)
    m = np.zeros((w.shape[0], seq_ref_1hot.shape[0], seq_ref_1hot.shape[1]))
    
    # prediction of reference sequence
    seqs_1hot_tf = tf.convert_to_tensor(seq_ref_1hot, dtype=tf.float32)[tf.newaxis]  
    latent_ref = new_model2(seqs_1hot_tf)
    latent_ref = tf.squeeze(latent_ref, axis=[0,2])

    # compute ISM
    for i in range(seq_ref_1hot.shape[0]):  
        out = []
        for j in range(4):
            tmp = np.copy(seq_ref_1hot)
            tmp[i,:] = [False, False, False, False]
            tmp[i,j] = True
            out += [tmp]
        
        out_tf = tf.convert_to_tensor(np.array(out), dtype=tf.float32)
        latent = new_model2(out_tf)
        latent = tf.squeeze(latent, axis=[2])
        latent = latent - latent_ref
        pred = tf.matmul(latent, w.transpose()) 
        m[:,i,:] = pred.numpy().transpose()
    return m

def calc_ism_from_bed(
    cell_embedding_ad:anndata.AnnData,
    peak_bed: Union[str, Path],
    fasta_file: Union[str, Path],
    XChrom_model:tf.keras.Model,
    output_path: Union[str, Path],
    cellembed_raw:str = 'X_pca_harmony',
    seq_len: int = 1344,
    save_individual: bool = True,
    **calc_ism_kwargs
    ):
    """
    Calculate the ISM from BED file.

    This function performs end-to-end ISM calculation starting from genomic coordinates
    in BED format. It extracts sequences, converts to one-hot encoding, and computes
    ISM matrices for all peaks.

    Parameters
    ----------
    cell_embedding_ad: anndata.AnnData
        anndata object with Initial cell embeddings
    cellembed_raw: str
        Key of the raw cell input embedding in the cell embedding adata,to generate model input.
    peak_bed: Union[str, Path]
        Path to BED file containing peak coordinates
    fasta_file: Union[str, Path]
        Path to genome FASTA file
    XChrom_model: tf.keras.Model
        XChrom model with trained weights
    output_path: Union[str, Path]
        Directory to save ISM results
    seq_len: int
        Sequence length, default 1344
    save_individual: bool
        Whether to save individual peak ISM files, default True
    **calc_ism_kwargs:
        Additional keyword arguments passed to calc_ism function
        
    Returns
    -------
    list
        List of ISM matrices for all peaks, each with shape (n_cells, seq_len, 4)
    
    Examples
    --------
    >>> ism_results = calc_ism_from_bed(
    ...     peak_bed='peaks.bed',
    ...     fasta_file='hg38.fa',
    ...     XChrom_model=trained_model,
    ...     output_path='./ISM_results/'
    ... )
    >>> print(f"Processed {len(ism_results)} peaks")
    >>> print(f"Each ISM matrix shape: {ism_results[0].shape}")
    
    Files Created
    -------------
    output_path/peakN_ism.npy : Individual ISM matrix for peak N (if save_individual=True)
    output_path/all_peaks_ism.npy : Combined ISM matrices for all peaks
    output_path/peak_coordinates.txt : Peak coordinates reference file
    """
    import os
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Extract sequences from BED file
    print("Extracting sequences from BED file...")
    seqs_dna, seqs_coords = make_bed_seqs(peak_bed, fasta_file=fasta_file, seq_len=seq_len)
    
    # Convert to one-hot encoding
    print("Converting to one-hot encoding...")
    seqs_ref_1hot = [dna_1hot(seq) for seq in seqs_dna]
    
    # Calculate ISM for each peak
    print(f"Calculating ISM for {len(seqs_ref_1hot)} peaks...")
    ism_results = []
    
    for i, seq_ref_1hot_i in enumerate(seqs_ref_1hot):
        print(f"Processing peak {i+1}/{len(seqs_ref_1hot)}")
        
        # Calculate ISM for this peak
        m = calc_ism(cell_embedding_ad, XChrom_model, seq_ref_1hot_i, cellembed_raw, **calc_ism_kwargs)
        ism_results.append(m)
        
        # Save individual peak ISM if requested
        if save_individual:
            np.save(f'{output_path}/peak{i}_ism.npy', m)
    
    # Save all results together
    all_ism = np.array(ism_results)  # shape: (n_peaks, n_cells, seq_len, 4)
    np.save(f'{output_path}/all_peaks_ism.npy', all_ism)
    print(f"All ISM matrices shape (n_peaks, n_cells, seq_len, 4): {all_ism.shape}")
    
    # Save peak coordinates for reference
    with open(f'{output_path}/peak_coordinates.txt', 'w') as f:
        for i, coord in enumerate(seqs_coords):
            if len(coord) == 3:
                f.write(f"peak{i}\t{coord[0]}\t{coord[1]}\t{coord[2]}\n")
            elif len(coord) == 4:
                f.write(f"peak{i}\t{coord[0]}\t{coord[1]}\t{coord[2]}\t{coord[3]}\n")
    
    print(f"ISM calculation completed. Results saved to {output_path}")
    return ism_results

def ism_norm(
    peak_ism,
    peak_bed,
    fasta_file,
    seq_len
    ):
    """
    Normalized the ISM scores for the four nucleotides at each position such that they summed to zero
    
    Parameters
    ----------
    peak_ism: np.ndarray
        ISM matrix, shape (n_peaks, n_cells, seq_len, 4)
    peak_bed: str
        Path to BED file containing peak coordinates
    fasta_file: str
        Path to genome FASTA file
    seq_len: int
        Sequence length, default 1344
        
    Returns
    -------
    np.ndarray
        Normalized ISM matrix, shape (n_peaks, n_cells, seq_len, 4)
    list
        List of peak coordinates
        
    Examples
    --------
    >>> ism_norm, seqs_coords = ism_norm(peak_ism, peak_bed, fasta_file, seq_len)
    >>> print(f"Normalized ISM matrix shape: {ism_norm.shape}")
    >>> print(f"Peak coordinates: {seqs_coords}")
    
    """
    seqs_dna, seqs_coords = make_bed_seqs(peak_bed,fasta_file=fasta_file,seq_len=seq_len)
    seq_ref_1hot = dna_1hot(seqs_dna[0])
    ref_scores = np.repeat(seq_ref_1hot[np.newaxis,:,:], peak_ism.shape[0], axis=0) * (peak_ism - np.repeat(peak_ism.mean(axis=2)[:,:,np.newaxis], 4, axis=2))
    ism_norm = ref_scores - np.repeat(ref_scores.mean(axis=0)[np.newaxis,:,:], peak_ism.shape[0], axis=0)
    return ism_norm,seqs_coords