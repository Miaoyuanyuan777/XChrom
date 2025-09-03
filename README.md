# XChrom：Cross-cell chromatin accessibility prediction model

<div align=left><img width = '700' height ='200' src ="XChrom_pipeline.png"/></div>

## Brief Introduction

**XChrom** is a deep learning model designed to predict chromatin accessibility across different cell types. Specifically, the model employs a **convolutional neural network (CNN)** architecture to extract sequence-level features for accessibility prediction, while simultaneously incorporating **cell identity information** to achieve generalization at the single-cell level. As a result, the model takes two distinct inputs:  

1. **One-hot encoded DNA sequences**, which provide the sequence information.  
2. **Cell identity vectors**, derived from dimensionality reduction of paired scRNA-seq data.  

Together, these inputs enable XChrom to predict whether a given sequence is accessible in specific cells.  

## Tutorial

https://xchrom.readthedocs.io/en/latest/

## Installation

XChrom is a deep learning project developed with TensorFlow 2.6.0 and Python 3.8, requiring GPU acceleration support. This documentation provides complete installation and configuration instructions.

If you haven't installed conda yet, please download and install Miniconda or Anaconda first.

### 1. Create and Activate Conda Environment

```bash
# Create a Python 3.8 environment named XChrom
conda create -n XChrom python=3.8

# Activate the environment
conda activate XChrom
```

### 2. Install Git and Clone the Project

```bash
# Install Git
conda install git

# Clone the project
git clone git@github.com:Miaoyuanyuan777/XChrom.git
```

### 3. Install TensorFlow GPU Version

Two installation methods are available (conda method is recommended):

**Method 1: Install with conda (Recommended)**

```bash
conda install tensorflow-gpu=2.6.0 -c conda-forge
```

**Method 2: Install with pip**

```bash
pip install tensorflow-gpu==2.6.0
```

### 4. Install CUDA and cuDNN

```bash
# Install CUDA 11.2 and cuDNN 8.1
conda install cudatoolkit=11.2 cudnn=8.1 -c conda-forge
```

After installation, run the following command to verify that TensorFlow can correctly recognize the GPU:

```python
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

If the output shows TensorFlow version 2.6.0 and detects GPU devices, the installation was successful.

### 5. Install Project Dependencies

```bash
# Navigate to the project directory
cd XChrom/

# Install all dependencies listed in requirements.txt
pip install -r requirements.txt

# Install XChrom
pip install .
```

## Quick start
```python
import xchrom as xc

data_path = xc.get_data_dir()
history = xc.tr.train_XChrom(
    input_folder = f'{data_path}/train_data',
    cell_embedding_ad = f'{data_path}/test_rna.h5ad',
    cellembed_raw='X_pca',
    out_path='./data/quick_start/train_out/',
    epochs = 10,
    verbose = 1
)

xc.pl.plot_train_history(
    history = history['history'],
    savefig = True,
    out_file = './data/quick_start/train_out/train_history_plot.pdf'
    )
```
