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

    conda install git
    git clone git@github.com:Miaoyuanyuan777/XChrom.git

    conda install tensorflow-gpu=2.6.0 -c conda-forge
    ## or
    pip install tensorflow-gpu==2.6.0

    ## use GPU
    conda install cudatoolkit=11.2 cudnn=8.1 -c conda-forge

    cd XChrom/
    pip install -r requirements.txt
    pip install .


Requirements of XChrom
----------------------

Those will be installed automatically when using pip.

::

    anndata==0.9.2
    biopython==1.79    
    ConfigArgParse==1.7
    logomaker
    matplotlib==3.5.1
    pandas==1.4.1
    pysam==0.23.3
    # rpy2==3.5.17
    scanpy==1.9.5
    scikit-learn==1.0.2
    scipy==1.8.0
    setuptools
    tqdm
    tensorflow-gpu==2.6.0 
    protobuf==3.20.3
    numpy
    h5py==3.1.0
    typing_extensions==3.7.4.3
    keras==2.6.0