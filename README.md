# XChrom

### Installation
```bash
conda create -n XChrom python=3.8
conda activate XChrom

conda install git
git clone git@github.com:Miaoyuanyuan777/XChrom.git
```

```
conda install -c conda-forge tensorflow-gpu=2.6.0
## or
pip install tensorflow-gpu==2.6.0
### if wrong, install it local
wget https://pypi.tuna.tsinghua.edu.cn/packages/79/78/561f7a29221a818f8dfd67d3bb45c64a3f8ecfdf082cec7e19a1d45839d0/tensorflow_gpu-2.6.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=80e68a0efba86eea87ea8ec9a1ffd1def48d22db16268af547305bf0ba889746
pip install tensorflow_gpu-2.6.0-cp38-cp38-manylinux2010_x86_64.whl
cd XChrom/xchrom
python setup.py build
python setup.py install
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
