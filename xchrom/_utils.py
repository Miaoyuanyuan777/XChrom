import sys
import random
import numpy as np
import pandas as pd
import pysam
from typing import Union
import tensorflow as tf
from pathlib import Path

def make_bed_seqs_from_df(
    input_bed: Union[str, pd.DataFrame],
    fasta_file: Union[str, Path],
    seq_len: int = 1344,
    stranded: bool = False
    ):
    """
    extract and expand BED sequences to specified length from DataFrame or file
    
    Parameters
    ----------
    input_bed: Union[str, pd.DataFrame]
        can be DataFrame(has chr,start,end columns) or bed file path
    fasta_file: Union[str, Path]
        genome fasta file path
    seq_len: int, default is 1344
        sequence length
    stranded: bool, default is False
        whether the sequence is stranded
    
    Returns
    -------
    seqs_dna: list
        extracted DNA sequence list
    seqs_coords: list
        sequence coordinate list
    """
    fasta_open = pysam.FastaFile(str(fasta_file))
    seqs_dna = []
    seqs_coords = []
    
    # process input(support file path or DataFrame)
    bed_entries = []
    
    # if it is a file path, read the file content
    if isinstance(input_bed, str) or isinstance(input_bed, Path):
        input_bed = Path(input_bed)
        with open(input_bed) as f:
            for line in f:
                a = line.split()
                # Skip empty lines
                if len(a) == 0:
                    continue
                chrm = a[0]
                start = int(float(a[1]))
                end = int(float(a[2]))
                strand = a[5] if len(a) >= 6 else "+"
                bed_entries.append((chrm, start, end, strand))
    else:
        # process DataFrame input
        for i in range(input_bed.shape[0]):
            chrm = input_bed.iloc[i,0]
            start = int(input_bed.iloc[i,1])
            end = int(input_bed.iloc[i,2])
            strand = "+"
            bed_entries.append((chrm, start, end, strand))
    
    # extract sequence
    for chrm, start, end, strand in bed_entries:
        # determine sequence boundary
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len
        
        # save coordinates
        if stranded:
            seqs_coords.append((chrm, seq_start, seq_end, strand))
        else:
            seqs_coords.append((chrm, seq_start, seq_end))
            
        # initialize sequence
        seq_dna = ""
        
        # process left out of range
        if seq_start < 0:
            print(
                "Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),
                file=sys.stderr,
            )
            seq_dna = "N" * (-seq_start)
            seq_start = 0
            
        # get DNA sequence
        seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()
        
        # process right out of range
        if len(seq_dna) < seq_len:
            print(
                "Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),
                file=sys.stderr,
            )
            seq_dna += "N" * (seq_len - len(seq_dna))
            
        # add to result list
        seqs_dna.append(seq_dna)
            
    fasta_open.close()
    return seqs_dna, seqs_coords

def make_bed_seqs(
    bed_file: str, 
    fasta_file: str, 
    seq_len: int = 1344, 
    stranded: bool = False
    ):
    """
    extract and expand sequence from bed file to specified length (compatibility function)
    
    Parameters
    ----------
    bed_file: str
        bed file path
    fasta_file: str
        genome fasta file path
    seq_len: int, default is 1344
        sequence length
    stranded: bool, default is False
        whether the sequence is stranded
    """
    return make_bed_seqs_from_df(bed_file, fasta_file, seq_len, stranded)

def dna_1hot(
    seq: str, 
    seq_len: int = None, 
    n_uniform: bool = False, 
    return_vec: bool = False
    ):
    """
    convert DNA sequence to one-hot encoding or vector representation
    
    Parameters
    ----------
    seq: str
        DNA sequence
    seq_len: int, default is None
        sequence length, if None, use the length of seq
    n_uniform: bool, default is False
        whether to represent N as 0.25 (force using float16), instead of random sampling
    return_vec: bool, default is False
        whether to return vector representation instead of one-hot encoding
        
    Returns
    -------
    seq_code: np.ndarray
        array representation of the sequence, if return_vec is True, return a 1D array, otherwise return a 2D array
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    if return_vec:
        # create 1D array representation
        seq_code = np.zeros((seq_len, ), dtype="int8")
        
        for i in range(seq_len):
            if i >= seq_start and i - seq_start < len(seq):
                nt = seq[i - seq_start]
                if nt == "A":
                    seq_code[i] = 0
                elif nt == "C":
                    seq_code[i] = 1
                elif nt == "G":
                    seq_code[i] = 2
                elif nt == "T":
                    seq_code[i] = 3
                else:
                    seq_code[i] = random.randint(0, 3)
    else:
        # create one-hot encoding representation
        if n_uniform:
            seq_code = np.zeros((seq_len, 4), dtype="float16")
        else:
            seq_code = np.zeros((seq_len, 4), dtype="bool")

        for i in range(seq_len):
            if i >= seq_start and i - seq_start < len(seq):
                nt = seq[i - seq_start]
                if nt == "A":
                    seq_code[i, 0] = 1
                elif nt == "C":
                    seq_code[i, 1] = 1
                elif nt == "G":
                    seq_code[i, 2] = 1
                elif nt == "T":
                    seq_code[i, 3] = 1
                else:
                    if n_uniform:
                        seq_code[i, :] = 0.25
                    else:
                        ni = random.randint(0, 3)
                        seq_code[i, ni] = 1
                        
    return seq_code

def dna_1hot_2vec(
    seq: str, 
    seq_len: int = None
    ):
    """
    convert DNA sequence to vector representation (compatibility function)
    
    Parameters
    ----------
    seq: str
        DNA sequence
    seq_len: int, default is None
        sequence length, if None, use the length of seq
    """
    return dna_1hot(seq, seq_len, return_vec=True)

def setup_seed(seed):
    np.random.seed(seed) 
    random.seed(seed)
    tf.random.set_seed(seed)
