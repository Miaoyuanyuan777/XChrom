.. highlight:: shell

============
Installation
============


XChrom is a deep learning project developed with TensorFlow 2.6.0 and Python 3.8, requiring GPU acceleration support. This documentation provides complete installation and configuration instructions.

If you haven't installed conda yet, please download and install Miniconda or Anaconda first.

**1. Create and Activate Conda Environment**

::

    # Create a Python 3.8 environment named XChrom
    conda create -n XChrom python=3.8

    # Activate the environment
    conda activate XChrom


**2. Install Git and Clone the Project**

::
    
    # Install Git
    conda install git
    
    # Clone the project
    git clone https://github.com/Miaoyuanyuan777/XChrom.git


**3. Install TensorFlow GPU Version**

Two installation methods are available (conda method is recommended):

(1) Install with conda (Recommend)

::
    
    conda install tensorflow-gpu=2.6.0 -c conda-forge



(2) Install with pip

::

    pip install tensorflow-gpu==2.6.0


**4. Install CUDA and cuDNN**

::

    # Install CUDA 11.2 and cuDNN 8.1
    conda install cudatoolkit=11.2 cudnn=8.1 -c conda-forge


After installation, run the following command to verify that TensorFlow can correctly recognize the GPU:

::

    python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"


If the output shows TensorFlow version 2.6.0 and detects GPU devices, the installation was successful.

**5. Install Project Dependencies**

::

    cd XChrom/

    # Install all dependencies listed in requirements.txt
    pip install -r requirements.txt

    # Install XChrom
    pip install .



Requirements of XChrom
----------------------

::

    anndata==0.9.2
    biopython==1.79    
    ConfigArgParse==1.7
    logomaker
    matplotlib==3.5.1
    pandas==1.4.1
    pysam==0.23.3
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