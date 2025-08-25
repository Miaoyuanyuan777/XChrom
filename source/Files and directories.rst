Files and directories
=====================

.. code-block:: text

    ├── 1_within_sample
    │   ├── e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.h5
    │   ├── eval_out
    │   │   ├── crossboth_percell_aucprc_scatterplot.pdf
    │   │   ├── crossboth_perpeak_aucprc_scatterplot.pdf
    │   │   ├── crossboth_pred.npy
    │   │   ├── crosscell_impute.h5ad
    │   │   ├── crosscell_percell_aucprc_scatterplot.pdf
    │   │   ├── crosscell_perpeak_aucprc_scatterplot.pdf
    │   │   ├── crosscell_pred.npy
    │   │   ├── crosscell_umap.pdf
    │   │   ├── crosspeak_percell_aucprc_scatterplot.pdf
    │   │   ├── crosspeak_perpeak_aucprc_scatterplot.pdf
    │   │   ├── crosspeak_pred.npy
    │   │   ├── denoise_impute.h5ad
    │   │   └── denoise_umap.pdf
    │   ├── m_brain_paired_atac.h5ad
    │   ├── m_brain_paired.h5ad
    │   ├── m_brain_paired_rna.h5ad
    │   ├── rna_pc32_lowRes.h5ad
    │   ├── train_data
    │   │   ├── ad_crosscell.h5ad
    │   │   ├── ad_crosspeak.h5ad
    │   │   ├── ad.h5ad
    │   │   ├── ad_test.h5ad
    │   │   ├── ad_trainval.h5ad
    │   │   ├── all_seqs.h5
    │   │   ├── m_crosscell.npz
    │   │   ├── m_crosspeak.npz
    │   │   ├── m_test.npz
    │   │   ├── m_trainval.npz
    │   │   ├── peaks.bed
    │   │   ├── peaks_test.bed
    │   │   ├── peaks_trainval.bed
    │   │   ├── splits.h5
    │   │   ├── test_seqs.h5
    │   │   └── trainval_seqs.h5
    │   ├── train_out
    │   │   ├── E1000best_CrossBothTrain_peakembedWB.h5
    │   │   ├── E1000best_model.h5
    │   │   ├── epoch_model
    │   │   ├── history.pickle
    │   │   └── train_history_plot.pdf
    ├── 2_cross_samples
    │   ├── eval_out
    │   │   ├── crosssamples_impute.h5ad
    │   │   ├── crosssamples_percell_aucprc_scatterplot.pdf
    │   │   ├── crosssamples_perpeak_aucprc_scatterplot.pdf
    │   │   ├── crosssamples_pred.npy
    │   │   └── crosssamples_umap.pdf
    │   ├── s1d1_s2d1.h5ad
    │   ├── test_atac.h5ad
    │   ├── test_data
    │   │   ├── ad.h5ad
    │   │   ├── all_seqs.h5
    │   │   ├── m.npz
    │   │   └── peaks.bed
    │   ├── test_rna_harmony.h5ad
    │   ├── train_atac.h5ad
    │   ├── train_data
    │   │   ├── ad_crosscell.h5ad
    │   │   ├── ad_crosspeak.h5ad
    │   │   ├── ad.h5ad
    │   │   ├── ad_test.h5ad
    │   │   ├── ad_trainval.h5ad
    │   │   ├── all_seqs.h5
    │   │   ├── m_crosscell.npz
    │   │   ├── m_crosspeak.npz
    │   │   ├── m_test.npz
    │   │   ├── m_trainval.npz
    │   │   ├── peaks.bed
    │   │   ├── peaks_test.bed
    │   │   ├── peaks_trainval.bed
    │   │   ├── splits.h5
    │   │   ├── test_seqs.h5
    │   │   └── trainval_seqs.h5
    │   ├── train_out
    │   │   ├── E1000best_model.h5
    │   │   ├── history.pickle
    │   │   └── train_history_plot.pdf
    │   └── train_rna_harmony.h5ad
    ├── 3_cross_species
    │   ├── eval_out
    │   │   ├── crosssamples_impute.h5ad
    │   │   ├── crosssamples_percell_aucprc_scatterplot.pdf
    │   │   ├── crosssamples_perpeak_aucprc_scatterplot.pdf
    │   │   ├── crosssamples_pred.npy
    │   │   └── crosssamples_umap.pdf
    │   ├── ISM_results
    │   │   ├── all_peaks_ism.npy
    │   │   ├── peak0_ism.npy
    │   │   └── peak_coordinates.txt
    │   ├── m1d1_atac.h5ad
    │   ├── m1d1_peaks.bed
    │   ├── m1d1_rna_harmony.h5ad
    │   ├── MEF2C.csv
    │   ├── mop3c2_atac.h5ad
    │   ├── mop3c2_rna_harmony.h5ad
    │   ├── test_data
    │   │   ├── ad.h5ad
    │   │   ├── all_seqs.h5
    │   │   ├── m.npz
    │   │   └── peaks.bed
    │   ├── train_data
    │   │   ├── ad_crosscell.h5ad
    │   │   ├── ad_crosspeak.h5ad
    │   │   ├── ad.h5ad
    │   │   ├── ad_test.h5ad
    │   │   ├── ad_trainval.h5ad
    │   │   ├── all_seqs.h5
    │   │   ├── m_crosscell.npz
    │   │   ├── m_crosspeak.npz
    │   │   ├── m_test.npz
    │   │   ├── m_trainval.npz
    │   │   ├── peaks.bed
    │   │   ├── peaks_test.bed
    │   │   ├── peaks_trainval.bed
    │   │   ├── splits.h5
    │   │   ├── test_seqs.h5
    │   │   └── trainval_seqs.h5
    │   └── train_out
    │       ├── E1000best_model.h5
    │       ├── epoch_model
    │       ├── history.pickle
    │       └── train_history_plot.pdf
    ├── 4_pred_newCondition
    │   ├── analysis_out
    │   │   └── tf_activity.h5ad
    │   ├── covid_rna_harmony.h5ad
    │   ├── Homo_sapiens.meme
    │   ├── motif_fasta
    │   │   ├── ref_peaks1000.fasta
    │   │   ├── shuffled_peaks.fasta
    │   │   ├── shuffled_peaks.h5
    │   │   └── shuffled_peaks_motifs
    │   │       ├── ALX1.fasta
    │   │       ├── ALX1.h5
    │   │       ├── ALX3.fasta
    │   │       ├── ...(other motifs)
    │   ├── pbmc10k_atac.h5ad
    │   ├── pbmc10k_rna_harmony.h5ad
    │   ├── train_data
    │   │   ├── ad_crosscell.h5ad
    │   │   ├── ad_crosspeak.h5ad
    │   │   ├── ad.h5ad
    │   │   ├── ad_test.h5ad
    │   │   ├── ad_trainval.h5ad
    │   │   ├── all_seqs.h5
    │   │   ├── m_crosscell.npz
    │   │   ├── m_crosspeak.npz
    │   │   ├── m_test.npz
    │   │   ├── m_trainval.npz
    │   │   ├── peaks.bed
    │   │   ├── peaks_test.bed
    │   │   ├── peaks_trainval.bed
    │   │   ├── splits.h5
    │   │   ├── test_seqs.h5
    │   │   └── trainval_seqs.h5
    │   └── train_out
    │       ├── E1000best_model.h5
    │       ├── epoch_model
    │       ├── history.pickle
    │       └── train_history_plot.pdf
    └── quick_start
        ├── train_data
        │   ├── ad_crosscell.h5ad
        │   ├── ad_crosspeak.h5ad
        │   ├── ad.h5ad
        │   ├── ad_test.h5ad
        │   ├── ad_trainval.h5ad
        │   ├── all_seqs.h5
        │   ├── m_crosscell.npz
        │   ├── m_crosspeak.npz
        │   ├── m_test.npz
        │   ├── m_trainval.npz
        │   ├── peaks.bed
        │   ├── peaks_test.bed
        │   ├── peaks_trainval.bed
        │   ├── splits.h5
        │   ├── test_seqs.h5
        │   └── trainval_seqs.h5
        └── train_out
            ├── E1000best_model.h5
            ├── epoch_model
            ├── history.pickle
            └── train_history_plot.pdf
