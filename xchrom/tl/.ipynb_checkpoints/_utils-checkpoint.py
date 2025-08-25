import os
from pathlib import Path
from typing import Union, Literal
import pandas as pd
import h5py
import anndata
import scanpy as sc
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from tqdm import tqdm
from .._utils import make_bed_seqs_from_df, dna_1hot_2vec


def calc_auc_pr(
    true_matrix:np.array, 
    pred_matrix:np.array, 
    mtype: Literal['overall', 'percell', 'perpeak'] = 'overall'
    ) -> dict:
    """
    Calculate the AUROC and AUPRC metrics for given true and predicted matrix.
    Shape of true_matrix and pred_matrix must be consistent, cells x peaks.
    
    Parameters
    ----------
    true_matrix: np.array
        The true label matrix (cells x peaks), 0/1 matrix, 0 for no signal, 1 for peak signal.
    pred_matrix: np.array,
        The predicted probability matrix (cells x peaks), 0-1 matrix for peak signal.
    mtype: Literal['overall', 'percell', 'perpeak'] = 'overall'
        The type of metric to calculate, optional 'overall', 'percell', 'perpeak'
    
    Examples
    --------
    >>> auc, prc = calc_auc_pr(true_matrix, pred_matrix, mtype='overall')
    >>> print(f"The AUROC is {auc:.4f}, The AUPRC is {prc:.4f}")
    """
    # check the shape of the matrix
    assert true_matrix.shape == pred_matrix.shape, "The shape of the true matrix and predicted matrix are not consistent"
    
    if mtype == 'overall':
        # calculate the overall metrics
        y_true = true_matrix.ravel()
        y_pred = pred_matrix.ravel()
        return {
            'auroc': roc_auc_score(y_true, y_pred),
            'auprc': average_precision_score(y_true, y_pred)
        }
    elif mtype == 'percell':
        # calculate the metrics per cell
        aurocs, auprcs = [], []
        for i in range(pred_matrix.shape[0]):
            y_true = true_matrix[i, :]
            # skip the cell with only one class
            if len(np.unique(y_true)) > 1:
                aurocs.append(roc_auc_score(y_true, pred_matrix[i, :]))
                auprcs.append(average_precision_score(y_true, pred_matrix[i, :]))
        return {
            'auroc': np.mean(aurocs) if aurocs else 0,
            'auprc': np.mean(auprcs) if auprcs else 0,
            'n_cells': len(aurocs)
        }
    elif mtype == 'perpeak':
        # calculate the metrics per peak
        aurocs, auprcs = [], []
        for j in range(pred_matrix.shape[1]):
            y_true = true_matrix[:, j]
            # skip the peak with only one class
            if len(np.unique(y_true)) > 1:
                aurocs.append(roc_auc_score(y_true, pred_matrix[:, j]))
                auprcs.append(average_precision_score(y_true, pred_matrix[:, j]))
        return {
            'auroc': np.mean(aurocs) if aurocs else 0,
            'auprc': np.mean(auprcs) if auprcs else 0,
            'n_peaks': len(aurocs)
        }
    else:
        raise ValueError("metrics type must be 'overall', 'percell' or 'perpeak'")

def calc_nsls_score(
    ad_rna:anndata.AnnData, 
    ad_atac:anndata.AnnData,
    n:int = 100,
    label:str = 'celltype', 
    test_cells:Union[list, np.ndarray] = None,
    use_rep_rna:str = 'X_pca', 
    use_rep_atac:str = 'X_pca'
    ):
    """
    Calculate the cluster metrics of scATAC data, including the number of shared neighbors and labels.
    
    Parameters
    ----------
    ad_rna: anndata.AnnData
        scRNA-seq data, used to calculate scRNA cell neighborhoods, which should have been processed with scanpy or others.
        Need to have raw cell represenation from scRNA-seq data, such as 'X_pca'.
    ad_atac: anndata.AnnData
        scATAC-seq raw or predicted data, used to calculate scATAC cell neighborhoods
        Must contain cell types, or clustering results from paired scRNA-seq data to calculate label scores
    n: int
        The number of neighbors in different scales, such as 100, 50, 10.
    label: str
        The key name of the cell type labels, default is 'celltype', or 'leiden' from scRNA-seq data.
        Should be assighed to ad_atac.obs[label]
    test_cells: list
        The cells to be computed, if None, all cells will be computed.
    use_rep_rna: str
        The key name of the scRNA cells dimension reduction, default is 'X_pca', to compute and generate scRNA neighbors list, which can be regarded as a genuine neighbor relationship.
    use_rep_atac: str
        The key name of the scATAC cells dimension reduction, default is 'X_pca' from scanpy(scanpy.tl.pca), to compute and generate scATAC neighbors list.
    
    Returns
    -------
    the ratio of shared neighbors: float
        The number of shared neighbors divided by the number of neighbors.
    the ratio of shared labels: float
        The number of shared labels divided by the number of neighbors.

    Data Requirements
    -----------------
    ad_rna.obsm must contain:
        - Cell dimension reduction (default: 'X_pca') for computing neighborhoods from scRNA-seq data
        
    ad_atac.obsm must contain:
        - Cell dimension reduction (default: 'X_pca') for computing neighborhoods from scRNA-seq data
        
    ad_atac.obs must contain:
        - Cell type labels (default: 'celltype') for label consistency evaluation,which can be from true cell type labels or paired scRNA-seq clustering results

    Examples
    --------
    >>> # Calculate the cluster metrics of scATAC data, including the number of shared neighbors and labels.
    >>> ns, ls = calc_nsls_score(ad_rna, ad_atac, n=100, label='celltype', test_cells=None, use_rep_rna='X_pca', use_rep_atac='X_pca')
    >>> print(f"The number of shared neighbors: {ns:.4f}, The number of shared labels: {ls:.4f}")
    """
    sc.pp.neighbors(ad_rna, n_neighbors=n+1, use_rep=use_rep_rna)
    m_RNA_neighbors = [i.indices for i in ad_rna.obsp['distances']]  # scRNA neighbors index
    
    sc.pp.neighbors(ad_atac, n_neighbors=n+1, use_rep=use_rep_atac)
    m_ATAC_neighbors = [i.indices for i in ad_atac.obsp['distances']]  # scATAC neighbors index
    
    # if test_cells is not None, only compute the cells in test_cells
    if test_cells is not None:
        # only keep the neighbors of test_cells
        test_RNA_neighbors = [m_RNA_neighbors[i] for i in test_cells]
        test_ATAC_neighbors = [m_ATAC_neighbors[i] for i in test_cells]
        
        # Calculate the number of shared neighbors
        n_shared_neighbors = np.mean([len(np.intersect1d(i, j)) for i, j in zip(test_RNA_neighbors, test_ATAC_neighbors)])
        
        # calculate the number of shared labels
        neighbor_label = ad_atac.obs[label].values[np.concatenate(test_ATAC_neighbors, axis=0)]
        cell_label = ad_atac.obs[label].values[np.repeat(test_cells, [len(m_ATAC_neighbors[i]) for i in test_cells])]
        n_shared_labels = (neighbor_label == cell_label).sum() / len(test_cells)
        
    else:
        # if test_cells is not None, use the global data
        n_shared_neighbors = np.mean([len(np.intersect1d(i, j)) for i, j in zip(m_RNA_neighbors, m_ATAC_neighbors)])
        neighbor_label = ad_atac.obs[label].values[np.concatenate(m_ATAC_neighbors, axis=0)]
        cell_label = ad_atac.obs[label].values[np.repeat(np.arange(len(m_ATAC_neighbors)), [len(j) for j in m_ATAC_neighbors])]
        n_shared_labels = (neighbor_label == cell_label).sum() / len(m_ATAC_neighbors)
    
    # calculate UMAP
    sc.tl.umap(ad_atac)
    
    return n_shared_neighbors/n, n_shared_labels/n

def calc_pca(
    ad:anndata.AnnData,
    max_value:float = None,
    n_comps:int = None
    ) -> anndata.AnnData:
    """
    Calculate the PCA of the given data, and save the PCA components in .obsm['X_pca'].

    Parameters
    ----------
    ad: anndata.AnnData
        The data to calculate the PCA.
    max_value: float = None
        The maximum value of the data.
    n_comps: int = None
        The number of components to calculate.

    Returns
    -------
    anndata.AnnData, With the PCA components in .obsm['X_pca'].

    Examples
    --------
    >>> ad = calc_pca(ad, max_value=10, n_comps=32)
    >>> print(f"The PCA components are saved in ad.obsm['X_pca']")
    """
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad)
    ad = ad[:, ad.var.highly_variable]
    sc.pp.scale(ad,max_value=max_value)
    sc.tl.pca(ad,n_comps = n_comps)
    return ad


def bed_to_fasta(
    bed_input:Union[str, Path, pd.DataFrame],
    fasta_file:Union[str, Path],
    output_file:Union[str, Path],
    seq_len:int = 1344,
    stranded:bool = False
    ):
    """
    Extract sequences from BED file and write to FASTA file.
    
    Parameters
    ----------
    bed_input: str, Path, or DataFrame
        The path to the BED file, or a pandas DataFrame with 'chr', 'start', 'end' columns.
    fasta_file: Union[str, Path]
        The path to the reference genome FASTA file.
    output_file: Union[str, Path]
        The path to the output FASTA file.
    seq_len: int, default 1344
        The length of the sequences to extract.
    stranded: bool, default False
        Whether to consider strand information.
        
    Returns
    -------
    tuple
        (seqs, coords) - The list of sequences and coordinates.
        
    Examples
    --------
    # Convert BED file to FASTA file
    seqs, coords = write_fasta("peaks.bed", "genome.fasta", "output.fasta", seq_len=1344)
    
    # Convert BED DataFrame to FASTA file
    seqs, coords = write_fasta(bed_df, "genome.fasta", "output.fasta", seq_len=1344)
    
    # Consider strand information
    seqs, coords = write_fasta("peaks.bed", "genome.fasta", "output.fasta", 
                               seq_len=1344, stranded=True)
    """
    
    print(f"Extracting sequences from BED file...")
    # print(f"BED input: {bed_input}")
    print(f"Reference genome: {fasta_file}")
    print(f"Sequence length: {seq_len}")
    print(f"Consider strand information: {stranded}")
    
    # Call make_bed_seqs_from_df function to extract sequences
    seqs, coords = make_bed_seqs_from_df(
        input_bed=bed_input,
        fasta_file=fasta_file,
        seq_len=seq_len,
        stranded=stranded
    )
    
    print(f"Successfully extracted {len(seqs)} sequences")
    
    # Write to FASTA file
    print(f"Writing to FASTA file: {output_file}")
    with open(output_file, 'w') as out:
        for coord, seq in zip(coords, seqs):
            if len(coord) == 3:
                chrom, start, end = coord
                header = f">{chrom}:{start}-{end}"
            elif len(coord) == 4:
                chrom, start, end, strand = coord
                header = f">{chrom}:{start}-{end}({strand})"
            else:
                # Fallback handling, if the coordinate format is not standard
                header = f">seq_{coords.index(coord)}"
            out.write(header + "\n")
            out.write(seq + "\n")
    
    print(f"FASTA file saved to: {output_file}")
    print(f"Number of sequences written: {len(seqs)}")
    return seqs, coords

def fasta_to_h5(
    input_fasta:Union[str, Path],
    seq_len:int = 1344
    ):
    """
    Encode the sequences in the FASTA file into HDF5 format.

    Parameters
    ----------
    input_fasta: Union[str, Path]
        The path to the FASTA file.
    seq_len: int
        The length of the sequences.

    Returns
    -------
    None
    """
    # Convert "motif.fasta" → "motif.h5"
    output_h5 = os.path.splitext(input_fasta)[0] + ".h5"
    records = list(SeqIO.parse(input_fasta, "fasta"))
    num_seqs = len(records)
    dna_array = np.zeros((num_seqs, seq_len), dtype="int8")
    for i, record in enumerate(records):
        seq = str(record.seq).upper()
        dna_array[i] = dna_1hot_2vec(seq, seq_len)
        
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("X", data=dna_array, dtype="int8")
        dt = h5py.string_dtype(encoding='utf-8')
        ids = [record.id for record in records]
        f.create_dataset("ids", data=np.array(ids, dtype=dt))

def generate_h5_files(
    motif_dir:Union[str, Path]
    ):
    """
    Generate HDF5 files from FASTA files.

    Parameters
    ----------
    motif_dir: Union[str, Path]
        The path to the directory containing the FASTA files.

    Returns
    -------
    tuple
        (h5_files, motif_names)
        
    Examples
    --------
    >>> h5_files, motif_names = generate_h5_files("motifs")
    >>> print(h5_files)
    >>> print(motif_names)
    """
    fasta_files = sorted([f for f in os.listdir(motif_dir) if f.endswith('.fasta')])
    h5_files = []
    motif_names = []
    
    for fasta_file in tqdm(fasta_files, desc="Generating H5 files"):
        fasta_path = os.path.join(motif_dir, fasta_file)
        h5_path = os.path.splitext(fasta_path)[0] + ".h5"
        try:
            fasta_to_h5(fasta_path)
            h5_files.append(h5_path)
            motif_names.append(os.path.splitext(fasta_file)[0])
        except Exception as e:
            print(f"Error processing {fasta_file}: {str(e)}")
            continue
    return h5_files, motif_names

def shuffle_sequences(sequences, k=2, seed=10):
    """
    Shuffle the sequences while keeping the k-mer frequency.
    Need to install fasta_ushuffle first.
    
    Parameters
    ----------
    sequences: list
        Input sequence list  
    k: int
        Length of k-mer to keep (default 2=dinucleotide frequency)
    seed: int
        Random seed
        
    Returns
    -------
    list: Shuffled sequences list
    """
    import subprocess
    import tempfile
    import os
    import shutil
    import sys
    
    # Try multiple ways to find fasta_ushuffle
    fasta_ushuffle_path = None
    
    # Method 1: Use shutil.which with current PATH
    fasta_ushuffle_path = shutil.which('fasta_ushuffle')
    
    # Method 2: Check current Python environment's bin directory
    if fasta_ushuffle_path is None:
        # Get the directory where current Python executable is located
        python_dir = os.path.dirname(sys.executable)
        potential_path = os.path.join(python_dir, 'fasta_ushuffle')
        if os.path.exists(potential_path) and os.access(potential_path, os.X_OK):
            fasta_ushuffle_path = potential_path
    
    # Method 3: Check CONDA_PREFIX environment variable
    if fasta_ushuffle_path is None:
        conda_env = os.environ.get('CONDA_PREFIX')
        if conda_env:
            potential_path = os.path.join(conda_env, 'bin', 'fasta_ushuffle')
            if os.path.exists(potential_path) and os.access(potential_path, os.X_OK):
                fasta_ushuffle_path = potential_path
    
    # Method 4: Try to infer conda environment from Python path
    if fasta_ushuffle_path is None:
        python_path = sys.executable
        # Check if we're in a conda environment (path contains /envs/ or /miniconda3/bin)
        if '/envs/' in python_path or '/miniconda3/bin' in python_path or '/anaconda3/bin' in python_path:
            # Extract environment root path
            if '/envs/' in python_path:
                env_root = python_path.split('/bin/python')[0]
            else:
                env_root = os.path.dirname(python_path)
            
            potential_path = os.path.join(env_root, 'fasta_ushuffle')
            if os.path.exists(potential_path) and os.access(potential_path, os.X_OK):
                fasta_ushuffle_path = potential_path
    
    # Method 5: Try common system paths as last resort
    if fasta_ushuffle_path is None:
        common_paths = [
            '/usr/local/bin/fasta_ushuffle',
            '/usr/bin/fasta_ushuffle',
            '/opt/conda/bin/fasta_ushuffle'
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                fasta_ushuffle_path = path
                break
    
    if fasta_ushuffle_path is None:
        raise Exception(
            f"fasta_ushuffle not found in current Python environment.\n"
            f"Current Python: {sys.executable}\n"
            f"Please install fasta_ushuffle in your current conda environment:\n"
            f"conda install bioconda::fasta_ushuffle"
        )
    
    print(f"Using fasta_ushuffle from: {fasta_ushuffle_path}")
    
    # create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq.upper()}\n")
        input_file = f.name

    try:
        # Use full path and inherit current environment
        cmd = f"{fasta_ushuffle_path} -k {k} -seed {seed} < {input_file}"
        
        # Get current environment and ensure PATH is properly set
        env = os.environ.copy()
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True,
            env=env,
            executable='/bin/bash'  # Explicitly use bash
        )
        
        # parse FASTA output
        shuffled_sequences = []
        current_seq = ""
        for line in result.stdout.strip().split('\n'):
            if line.startswith('>'):
                if current_seq:
                    shuffled_sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            shuffled_sequences.append(current_seq)
            
        return shuffled_sequences
        
    except subprocess.CalledProcessError as e:
        raise Exception(f"fasta_ushuffle failed: {e.stderr}. Install with: conda install bioconda::fasta_ushuffle")
    except FileNotFoundError:
        raise Exception("fasta_ushuffle not found. Please install it first: conda install bioconda::fasta_ushuffle")
    finally:
        try:
            os.unlink(input_file)
        except:
            pass

def read_meme_motifs(meme_file):
    """
    Read MEME format motif file
    
    Parameters
    ----------
    meme_file: str
        MEME format motif file path
        
    Returns
    -------
    motifs: list
        motif information list, each element is a dict containing motif name and pwm
    """
    motifs = []
    current_motif = None
    pwm_lines = []
    reading_pwm = False
    
    with open(meme_file, 'r') as f:
        for line in f:
            line = line.strip() # remove leading and trailing whitespace
            
            if line.startswith('MOTIF'):
                # save previous motif
                if current_motif is not None and pwm_lines:
                    pwm_matrix = np.array([list(map(float, line.split())) for line in pwm_lines])
                    current_motif['pwm'] = pwm_matrix
                    motifs.append(current_motif)
                
                # start new motif
                parts = line.split() # default split by whitespace (including space, tab, newline, etc.)
                if len(parts) >= 2:
                    motif_name = parts[2]
                    # clean motif name, remove parentheses
                    motif_name = motif_name.replace('(', '').replace(')', '')
                    if '_' in motif_name:
                        motif_name = motif_name.split('_')[0]
                    current_motif = {'name': motif_name}
                    pwm_lines = []
                    reading_pwm = False
            
            elif line.startswith('letter-probability matrix'):
                reading_pwm = True
                pwm_lines = []
            
            elif reading_pwm and line and not line.startswith('URL'):
                # check if it is a number line
                try:
                    values = list(map(float, line.split()))
                    if len(values) == 4:  # A, C, G, T
                        pwm_lines.append(line)
                except:
                    reading_pwm = False
    
    # process the last motif
    if current_motif is not None and pwm_lines:
        pwm_matrix = np.array([list(map(float, line.split())) for line in pwm_lines])
        current_motif['pwm'] = pwm_matrix
        motifs.append(current_motif)
    
    return motifs

def generate_motif_sequences(pwm, n_sequences=1000, seed=10):
    """
    Generate motif sequences based on PWM.
    
    Parameters
    ----------
    pwm: np.array
        Position weight matrix (length x 4), 4 columns correspond to A,C,G,T
    n_sequences: int
        Number of sequences to generate
    seed: int
        Random seed
        
    Returns
    -------
    sequences: list
        Generated motif sequences list
    """
    np.random.seed(seed)
    bases = ['A', 'C', 'G', 'T']
    motif_seqs = []
    
    # Normalize PWM to ensure each row sums to 1 (fix floating point precision issues)
    pwm_normalized = pwm / pwm.sum(axis=1, keepdims=True)
    
    for _ in range(n_sequences):
        seq = ''
        for pos in range(pwm_normalized.shape[0]):
            # select base based on PWM probability
            base_idx = np.random.choice(4, p=pwm_normalized[pos])
            seq += bases[base_idx]
        motif_seqs.append(seq)
    
    return motif_seqs

def insert_motif_to_background(background_seqs, motif_seqs):
    """
    Insert motif sequences into the center of background sequences
    
    Parameters
    ----------
    background_seqs: list
        background sequences list
    motif_seqs: list
        motif sequences list
        
    Returns
    -------
    inserted_seqs: list
        Inserted sequences list
    """
    inserted_seqs = []
    motif_len = len(motif_seqs[0])
    
    for i, bg_seq in enumerate(background_seqs):
        if i < len(motif_seqs):
            # calculate insertion position
            seq_len = len(bg_seq)
            left_coord = seq_len // 2 - motif_len // 2
            
            # 插入motif
            left_part = bg_seq[:left_coord]
            right_part = bg_seq[left_coord + motif_len:]
            inserted_seq = left_part + motif_seqs[i] + right_part
            inserted_seqs.append(inserted_seq)
        else:
            # if motif sequence is not enough, repeat use
            motif_idx = i % len(motif_seqs)
            seq_len = len(bg_seq)
            left_coord = seq_len // 2 - motif_len // 2
            
            left_part = bg_seq[:left_coord]
            right_part = bg_seq[left_coord + motif_len:]
            inserted_seq = left_part + motif_seqs[motif_idx] + right_part
            inserted_seqs.append(inserted_seq)
    
    return inserted_seqs

def generate_tf_activity_data(
    bed_file: Union[str, Path],
    input_fasta: Union[str, Path],
    motif_file: Union[str, Path],
    output_dir: Union[str, Path],
    n_samples: int = 1000,
    seq_len: int = 1344,
    n_motif_instances: int = 1000,
    seed: int = 10
    ):
    """
    Prepare motif data and background sequences for TF activity calculation
    
    Parameters
    ----------
    bed_file: Union[str, Path]
        BED file path, containing peak regions
    input_fasta: Union[str, Path]
        Reference genome FASTA file path
    motif_file: Union[str, Path]
        MEME format motif file path
    output_dir: Union[str, Path]
        Output directory path for the generated data
    n_samples: int, default 1000
        Number of sampled peaks
    seq_len: int, default 1344
        Sequence length
    n_motif_instances: int, default 1000
        Number of instances to generate for each motif
    seed: int, default 10
        Random seed
        
    Returns
    -------
    tuple
        (background_fasta_path, motif_dir_path) - background sequence file path and motif directory path
        
    Examples
    --------
    >>> bg_fasta, motif_dir = prepare_motif_data(
    ...     bed_file="peaks.bed",
    ...     input_fasta="hg38.fa", 
    ...     motif_file="motifs.meme",
    ...     output_dir="./motif_fasta",
    ...     n_samples=1000,
    ...     seed=10
    ... )
    """
    
    print("=== Start preparing motif data ===")
    
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Read BED file and sample
    print(f"1. Read BED file: {bed_file}")
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, 
                        names=['chr', 'start', 'end'] + [f'col{i}' for i in range(3, 10)])
    
    # adjust region size to 1344bp, center
    bed_df['center'] = (bed_df['start'] + bed_df['end']) // 2
    bed_df['start'] = bed_df['center'] - seq_len // 2  
    bed_df['end'] = bed_df['center'] + seq_len // 2
    bed_df = bed_df[['chr', 'start', 'end']]
    
    # random sampling
    np.random.seed(seed)
    if len(bed_df) > n_samples:
        sampled_indices = np.random.choice(len(bed_df), n_samples, replace=False)
        bed_df = bed_df.iloc[sampled_indices].reset_index(drop=True)
    
    print(f"Sampled {len(bed_df)} regions from {bed_file}")
    
    # 2. Extract background sequences
    print("2. Extract background sequences...")
    ref_fasta_path = output_dir / "ref_peaks1000.fasta"
    seqs, _ = bed_to_fasta(bed_df, input_fasta, ref_fasta_path, seq_len)
    print(f"Extracted {len(seqs)} sequences, saved to {ref_fasta_path}")
    
    # 3. Generate shuffled background sequences
    print("3. Generate shuffled background sequences...")
    shuffled_seqs = shuffle_sequences(seqs, seed=seed)
    background_fasta_path = output_dir / "shuffled_peaks.fasta"
    with open(background_fasta_path, 'w') as f:
        for i, seq in enumerate(shuffled_seqs):
            f.write(f">shuffled_seq_{i}\n")
            f.write(seq + "\n")
    print(f"Dinucleotide shuffled {len(shuffled_seqs)} sequences, saved to {background_fasta_path}")
    
    # 4. Read motif file (meme format)
    print("4. Read motif file(meme format)...")
    motifs = read_meme_motifs(motif_file) # list of dicts, each dict contains motif name and pwm
    print(f"Read {len(motifs)} motifs from {motif_file}")
    
    # 5. Generate motif inserted sequences
    print("5. Generate motif inserted sequences...")
    motif_output_dir = output_dir / "shuffled_peaks_motifs"
    motif_output_dir.mkdir(exist_ok=True)
    
    for motif in tqdm(motifs, desc="Processing motifs"):
        try:
            # generate motif sequence instances
            motif_instances = generate_motif_sequences(
                motif['pwm'], n_motif_instances, seed=seed
            )
            
            # insert into background sequences
            inserted_seqs = insert_motif_to_background(shuffled_seqs, motif_instances)
            
            # write to fasta file
            motif_fasta_path = motif_output_dir / f"{motif['name']}.fasta"
            with open(motif_fasta_path, 'w') as f:
                for i, seq in enumerate(inserted_seqs):
                    f.write(f">motif_{motif['name']}_{i}\n")
                    f.write(seq + "\n")                    
        except Exception as e:
            print(f"Error processing motif {motif['name']}: {str(e)}")
            continue
    
    print("=== motif data preparation completed ===")
    print(f"Background sequence file: {background_fasta_path}")
    print(f"Motif inserted sequence directory: {motif_output_dir}")
    
    return str(background_fasta_path), str(motif_output_dir)
