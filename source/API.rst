API
~~~

XChrom: Cross-cell chromatin accessibility prediction model.
============================================================


Import xchrom as::

   import xchrom as xc


Reading
=======

.. autosummary::
   :toctree: _autosummary

   xchrom.readfile.read_10x_mtx_to_adata
   xchrom.readfile.read_rds_to_h5ad
   xchrom.readfile.read_h5seurat_to_h5ad
   xchrom.readfile.read_seurat_to_h5ad
   xchrom.readfile.read_10x_h5_to_h5ad


Preprocessing
=============

.. autosummary::
   :toctree: _autosummary

   xchrom.pp.process_train_test_single
   xchrom.pp.process_test_dual

Model training
==============

.. autosummary::
   :toctree: _autosummary

   xchrom.tr.train_XChrom
   xchrom.tr.Generator
   xchrom.tr.XChrom_model


Toolkit
=======

.. autosummary::
   :toctree: _autosummary

   xchrom.tl.calc_auc_pr
   xchrom.tl.calc_nsls_score
   xchrom.tl.calc_pca
   xchrom.tl.crosscell_aucprc
   xchrom.tl.crosscell_nsls
   xchrom.tl.crosspeak_aucprc
   xchrom.tl.crossboth_aucprc
   xchrom.tl.denoise_nsls
   xchrom.tl.crosssamples_aucprc
   xchrom.tl.crosssamples_nsls
   xchrom.tl.calc_ism
   xchrom.tl.ism_norm
   xchrom.tl.calc_ism_from_bed
   xchrom.tl.generate_tf_activity_data
   xchrom.tl.calc_tf_activity

Plotting
========

.. autosummary::
   :toctree: _autosummary

   xchrom.pl.plot_train_history
   xchrom.pl.plot_logo
   xchrom.pl.plot_percell_aucprc
   xchrom.pl.plot_perpeak_aucprc
   xchrom.pl.plot_motif_activity
