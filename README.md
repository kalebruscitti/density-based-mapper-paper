## Density-Based Mapper Paper
This repository holds python code and Jupyter notebooks used to generate the figures in the paper
`Density-Adaptive Mapper Covers for Temporal Topic Modelling`

There are two subfolders, `ungdc` contains notebooks related to temporal topic modelling of the United
Nations general debate corpus, and `synthetic-data` contains code for the computational experiments on
synethetically generated data.

### Temporal-Mapper Package Repository
The implementation of density-based Mapper is in another repository:

    https://www.github.com/tutteinstitute/temporal-mapper

This implementation is also uploaded to PyPI as the `temporal-mapper` package, and it is this package 
which is imported and used in this repository.


### Running the United Nations modelling code
The input data for the UNGDC is too large to host on GitHub, so you will need to fetch it. It is available at on Kaggle, at:

    https://www.kaggle.com/datasets/unitednations/un-general-debates

After downloading the dataset, you should run the notebook `UNGDC-Compute.ipynb` which does the initial processing of the data. 
This includes chunking the transcripts, embedding and dimensional reduction. Once that is completed, it will save the results
to a parquet file that can be loaded in by the other notebooks.

### Running the synthetic data code
The synthetic data used to generate the figures in the paper is included in the repo, `genus1_demo.npy`. The process used to
generate this data is shown in `datagen.ipynb`. You may run `g1test.ipynb` to perform the experiments in the paper and graph the results.
