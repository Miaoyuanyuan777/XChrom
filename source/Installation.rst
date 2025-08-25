.. highlight:: shell

============
Installation
============


Install by PyPi
---------------

**Step 1:**

Prepare conda environment for XChrom:
::

	conda create -n XChrom python=3.8
	conda activate XChrom

**Step 2:**

Install XChrom using `pip`:
::

	pip install xchrom


Install by github
-----------------

Download the file from github:
::

    cd XChrom
    python setup.py build
    python setup.py install


Requirements of XChrom
----------------------

Those will be installed automatically when using pip.

::

    anndata==0.9.2
    biopython==1.79
    h5py==3.1.0
    matplotlib==3.5.1
    numpy==1.19.2
    pandas==1.4.1
    pysam==0.9.1
    rpy2==3.5.17
    scanpy==1.9.5
    scikit-learn==1.0.2 
    scipy==1.8.0
    setuptools==68.2.2
    tensorflow==2.6.0
    tqdm==4.66.1

Install tensorflow with GPU support : conda install -c conda-forge tensorflow-gpu=2.6.0