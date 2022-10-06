# Topyfic

## Topyfic: Reproducible latent dirichlet allocation (LDA) using leiden clustering and harmony for single cell epigenomics data

An open challenge for the analysis of single-cell data is the identification of distinct cellular programs that may be simultaneously expressed in the same cell based on the interaction of genotypes in environments. Latent Dirichlet allocation (LDA) is a popular statistical method for the identification of recurring patterns in count data (e.g. gene expression), which are referred to as topics. These topics are composed of genes with specific weights that can together explain underlying patterns of gene expression profile for each individual cell. In particular, each cell’s expression profile can be decomposed into a combination of the topics that can be analyzed both globally using topic-trait enrichment as well as in individual cells using structure plots. Due to the random initialization of LDA algorithms, topic definitions can vary substantially each time that the algorithm is rerun, which hinders their interpretability. Therefore, we developed reproducible LDA where we define our topics by analysing their reproducibility across a large number of runs.

Topyfic is a Python library designed to apply rLDA to single_cells/bulk RNA-seq data to recover meaningful topics that involve the key transcription factors involved in myogenic differentiation.

![LDA overview](docs/TopicModels.png)

![Topyfic overview](docs/Topyfic.png)

## Documentation
Topyfic's full documentation can be found at [here](https://mortazavilab.github.io/Topyfic/html/index.html)

## Installation

To install Topyfic, python version 3.8 or greater is required.

### Install from PyPi (recommended)
Install the most recent release, run

`pip install Topyfic`

### Install with the most recent commits
git cloning the [Topyfic repository](https://github.com/mortazavilab/Topyfic), going to the Topyfic directory, run

`pip install .`

## Tutorials

comming soon ....

## Cite

comming soon ...
