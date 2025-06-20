# Introduction

SpaDicer is a deep learning framework that unifies single-cell RNA sequencing and spatial transcriptomics data, combining pseudo-spatial reconstruction and cell-type deconvolution to enhance spatial localization and cell-type composition analysis of complex tissue.	

# Model architecture

![model](./model.png)

# Requirements

The deep learning models were trained on 2*NVIDIA GeForce RTX 4090 on linux.

+ Python 3.8.12
+ CUDA 11.3.1
+ PyTorch 1.0
+ Pandas 1.5.0
+ Numpy 1.24.4
+ Scikit-learn 1.1.2

# Installation Guide

```
1.Clone the SpaDicer repository
```

`git clone https://github.com/hliulab/SpaDicer.git`

```
2.Enter SpaDicer project folder
```

`cd SpaDicer/`

```
3.Set up the Python environment and install the required packages
```

`conda create --name <your_env_name> --file requirements.txt`


# Directory structure

+ `model`: contains the code for the model, the evaluation.
+ `configs.py`: Configuration for hyperparameters.
+ `data`: This directory needs to be created by yourself to store experimental data and includes data preprocessing scripts.
+ `utils.py`:Tools and methods required for training or validation.
+ `train.py`:Training script.
+ `val.py`:validation script.
+ `dataset.py`:Load training and validation dataset scripts.

# Usage

First, you need to run the process.py script in the data directory to preprocess the single-cell RNA sequencing data and spatial transcriptome data.

```python
python spatial_processing.py --sc_data ./STARmap/sc_data.csv --sc_meta ./STARmap/sc_meta.csv --st_data ./STARmap/st_data.csv --st_meta ./STARmap/st_meta.csv --st_type spot --n_features 2000 --normalize --select_hvg union --output_sc_csv_path ./STARmap/sc_data_output.csv --output_st_csv_path ./STARmap/st_data_output.csv
```

- `--sc_data`: Path to the scRNA-seq data file (CSV format).
- `--sc_meta`: Path to the metadata for scRNA-seq.
- `--st_data`: Path to the spatial transcriptomics data file (CSV format).
- `--st_meta`: Path to the metadata for spatial transcriptomics.
- `--output_sc_csv_path`:Path to the scRNA-seq data output file.
- `--output_st_csv_path`:Path to the spatial transcriptomics data output file.
- `--st_type`: Type of spatial transcriptomics data, either `spot` or `image`.
- `--n_features`: Number of highly variable genes to select.
- `--normalize`: Flag to normalize the data.
- `--select_hvg`: Method to select highly variable genes, either `intersection` or `union`.
  
When dissecting intratumor heterogeneity of tumor tissues, it is necessary to perform clustering on SCC and BC by executing the following commands.

```shell
python spatial_analysis.py cluster \
    --expression_paths slice1_expr.csv slice2_expr.csv slice3_expr.csv \
    --spatial_paths slice1_spatial.csv slice2_spatial.csv slice3_spatial.csv \
    --output_paths slice1_clusters.csv slice2_clusters.csv slice3_clusters.csv \
    --n_clusters 5
```

- `--expression_paths`: Three expression file paths separated by spaces.
- `--spatial_paths`: Three spatial data file paths separated by spaces.
- `--output_paths`: Three output file paths separated by spaces.

To reproduce our experiments, you need to download the dataset (Example: HEART), and place it in the datasets folder. Next, you need to modify the `config.py` file to specify the path of dataset and change the hyperparameters as below:

```python
inf = 610								# Input feature dimensions
input_class = 8							# Reference number of single cell classifications
lr = 1e-3								# Learning Rate
n_epoch = 200							# The number of epochs
batchsize=64							# The number of samples processed before the model’s internal parameters are updated. 					
weight_decay = 1e-6						#  A regularization term added to the loss function to prevent overfitting by penalizing large weights.
sc_data = r'./data/HEART/sc_data.csv' # The path to the single-cell RNA-seq data file (in CSV format) used for training the model.
sc_meta = r'./data/HEART/sc_meta.csv' # The path to the metadata file for the single-cell RNA-seq data
st_data = r'./data/HEART/st_data.csv'	# The path to the spatial transcriptomics data file (in CSV format) used for training the model.
st_meta = r'./data/HEART/st_meta.csv'	# The path to the metadata file for the spatial transcriptomics data
```

To train the model and evaluate its performance on the test set, please run the following commands:
```shell
python train.py
python val.py
```



