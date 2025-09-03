.. XChrom documentation master file, created by
   sphinx-quickstart on Wed Jul 23 19:16:06 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XChrom's documentation!
==================================
**XChrom** is a deep learning model designed to predict chromatin accessibility across different cell types. Specifically, the model employs a **convolutional neural network (CNN)** architecture to extract sequence-level features for accessibility prediction, while simultaneously incorporating **cell identity information** to achieve generalization at the single-cell level. As a result, the model takes two distinct inputs:  

1. **One-hot encoded DNA sequences**, which provide the sequence information.  
2. **Cell identity vectors**, derived from dimensionality reduction of paired scRNA-seq data.  

Together, these inputs enable XChrom to predict whether a given sequence is accessible in specific cells. 


.. image:: ./_static/XChrom_pipeline.png
    :alt: Title figure
    :width: 700px
    :align: center

.. toctree::
   :maxdepth: 1
   :caption: Overview
   
   Installation
   Quickstart
   Files and directories
   
.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   Tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API

   API  


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
