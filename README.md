# XChrom

### Installation
```bash
conda create -n XChrom python=3.8
conda activate XChrom

conda install git
git clone git@github.com:Miaoyuanyuan777/XChrom.git
```

```
conda install tensorflow-gpu=2.6.0 -c conda-forge
## or
pip install tensorflow-gpu==2.6.0

## use GPU
conda install cudatoolkit=11.2 cudnn=8.1 -c conda-forge

cd XChrom/
pip install -r requirements.txt
pip install .

```

### Quick start
```
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

### Tutorials
see: 
```https://xchrom.readthedocs.io/en/latest/Tutorials/index.html```
