"""
Caculating evaluation metrics and analyzing the model results.

This module contains functions for model evaluation and analysis including:
- Calculate metrics, eg. AUROC, AUPRC, etc.
- Motif analysis and activity scoring
- In Silico Mutagenesis (ISM) analysis

"""

from ._utils import (
    calc_auc_pr,
    calc_nsls_score,
    calc_pca,
    bed_to_fasta,
    generate_tf_activity_data
)
from ._eval import (
    crosscell_aucprc,
    crosscell_nsls,
    crosspeak_aucprc,
    crossboth_aucprc,
    denoise_nsls,
    crosssamples_aucprc,
    crosssamples_nsls,
)
from ._ism import (
    calc_ism,
    calc_ism_from_bed,
    ism_norm
)
from ._tf_activity import (
    calc_tf_activity
)

__all__ = [
    'calc_auc_pr',
    'calc_nsls_score', 
    'calc_pca',
    'bed_to_fasta',
    'generate_tf_activity_data',
    'crosscell_aucprc',
    'crosscell_nsls',
    'crosspeak_aucprc',
    'crossboth_aucprc',
    'denoise_nsls',
    'crosssamples_aucprc',
    'crosssamples_nsls',
    'calc_ism',
    'calc_ism_from_bed',
    'ism_norm',
    'calc_tf_activity'
]
