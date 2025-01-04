# UrbanSparse

- This is the official repository of the paper:  *Reconciling Geospatial Prediction and Retrieval via Sparse Representations*

## Quick Start

- With this repository, you can
  - Download the Bloom filters and the (anonymized) locations of Beijing and Shanghai datasets.
  - Quickly reproduce all effectiveness results in prediction and retrieval tasks.
  - Compare with baselines DGI and MVGRL.

### Environment Preparation

- Please use **Miniconda or Anaconda**
- We use Python 3.11.5. Lower versions are not tested.

  - conda create -n UrbanSparse python==3.11.5

We require the following packages:

- PyTorch>=2.1.1
- cuda-cudart (install via **conda install -c conda-forge cuda-cudart**. This is for the custom CUDA kernel compilation.)
- jieba_fast
- tqdm
- scikit-learn==1.4.2
- xxhash==3.4.1
- pytorch-geometric (latest version), and its dependencies including torch-scatter and torch-sparse

## Experiments

- We provide scripts to repeat the experiments in the paper. 
- Please run the scripts in the scripts/ folder.

### Emebeddings for baselines

- The raw Beijing and Shanghai datasets are not provided due to license constraints.
- However, we provide the OpenAI, BERT, and trained DPR embeddings in Beijing and Shanghai so you can fully reproduce the results in the paper. [Google Drive](https://drive.google.com/drive/folders/1GeB7A90cocWvUJGysVxyK1pqyHKcfX5B?usp=drive_link)
- Please store the embeddings in the baselines/BERT/embeddings/ folder. (See baselines/GraphMethods/*.py)

## Thanks for reading!
