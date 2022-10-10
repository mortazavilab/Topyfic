import sys

import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import warnings
import joblib
import os
from multiprocessing import Pool
from itertools import repeat
import pickle
from sklearn.decomposition import LatentDirichletAllocation
import scanpy.external as sce

from Topyfic.train import *
from Topyfic.topic import *
from Topyfic.topModel import *

warnings.filterwarnings("ignore")


def calculate_leiden_clustering(trains, data, n_top_genes=50, resolution=1):
    """
    Do leiden clustering w/o harmony base on number of assays you have and then remove low participation topics
    :param trains: list of train class
    :type trains: list of Train
    :param data: gene-count data with cells and genes information
    :type data: anndata
    :param n_top_genes: Number of highly-variable genes to keep (default: 50)
    :type n_top_genes: int
    :param resolution: A parameter value controlling the coarseness of the clustering. Higher values lead to more clusters. (default: 1)
    :type resolution: int
    :return: final topmodel instance after clustering and trimming, dataframe containing which run goes to which topic
    :rtype: TopModel, pandas dataframe
    """
    all_batches = None
    all_components = None
    all_exp_dirichlet_component = None
    all_others = None

    for train in trains:
        components, exp_dirichlet_component, others = train.make_LDA_models_attributes()
        components.index = f"{train.name}_" + components.index
        exp_dirichlet_component.index = f"{train.name}_" + exp_dirichlet_component.index
        others.index = f"{train.name}_" + others.index
        batch = [train.name] * components.shape[0]
        if all_components is None:
            all_batches = batch
            all_components = components
            all_exp_dirichlet_component = exp_dirichlet_component
            all_others = others
        else:
            all_batches = batch + all_batches
            all_components = pd.concat([components, all_components], axis=1)
            all_exp_dirichlet_component = pd.concat([exp_dirichlet_component, all_exp_dirichlet_component], axis=1)
            all_others = pd.concat([others, all_others], axis=1)

    if len(trains) == 1:
        adata = anndata.AnnData(all_components)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
        sc.pl.umap(adata, color=["leiden"],
                   title=[f"Topic space UMAP leiden clusters (k={trains[0].k})"],
                   save="_leiden_clustring.pdf")

    else:
        adata = anndata.AnnData(all_components)
        adata.obs['assays'] = all_batches
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        sc.pp.neighbors(adata)
        sce.pp.harmony_integrate(adata, 'assays')
        adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
        sc.pp.neighbors(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
        sc.pl.umap(adata, color=['leiden'],
                   title=[f'Topic space UMAP leiden clusters'],
                   save='_leiden_clustring_harmony.png')
        sc.pl.umap(adata, color=['assays'],
                   title=['Topic space UMAP leiden clusters'],
                   save='_technology_harmony.png')

    clustering = adata.obs.copy(deep=True)
    n_rtopics = len(np.unique(clustering[f"leiden"]))

    rlda = initialize_rLDA_model(all_components,
                                 all_exp_dirichlet_component,
                                 all_others,
                                 clusters=clustering)
    lda_output = rlda.transform(data.X)
    cell_participation = pd.DataFrame(np.round(lda_output, 2),
                                      columns=[f"Topic{i + 1}" for i in range(n_rtopics)],
                                      index=data.obs.index)

    keep = cell_participation.sum() > data.to_df().shape[0] / 10000
    print(f"{keep.sum()} topics our of {keep.shape[0]} topics have participation in more than {data.to_df().shape[0] / 10000}")
    n_rtopics = keep.sum()
    rlda, gene_weights_T = filter_LDA_model(rlda, keep)

    gene_weights = gene_weights_T.T
    gene_weights.index = data.var.index

    if len(trains) == 1:
        top_model = TopModel(name=trains[0].name,
                             N=n_rtopics,
                             gene_weights=gene_weights,
                             rlda=rlda)

    else:
        name = np.unique(all_batches).tolist()
        name = '_'.join(name)

        top_model = TopModel(name=name,
                             N=n_rtopics,
                             gene_weights=gene_weights,
                             rlda=rlda)

    return top_model, clustering


def initialize_rLDA_model(all_components, all_exp_dirichlet_component, all_others, clusters):
    """
    Initialize reproducible LDA model by calculating all necessary attributes using clustering.
    :param all_components: Variational parameters for topic gene distribution from all single LDA models
    :type all_components: pandas dataframe
    :param all_exp_dirichlet_component: Exponential value of expectation of log topic gene distribution from all single LDA models
    :type all_exp_dirichlet_component: pandas dataframe
    :param all_others: dataframe contains remaining necessary attributes including
            n_batch_iter: Number of iterations of the EM step.
            n_features: Number of features seen during fit.
            n_iter: Number of passes over the dataset.
            bound: Final perplexity score on training set.
            doc_topic_prior: Prior of document topic distribution theta. If the value is None, it is 1 / n_components.
            topic_word_prior: Prior of topic word distribution beta. If the value is None, it is 1 / n_components.
    :type all_others: pandas dataframe
    :param clusters: dataframe that mapped each LDA run to each clusters
    :type clusters: pandas dataframe
    :return: Latent Dirichlet Allocation with online variational Bayes algorithm.
    :rtype: sklearn.decomposition.LatentDirichletAllocation
    """
    n_rtopics = len(np.unique(clusters[f"leiden"]))

    components = np.zeros((n_rtopics, all_components.shape[1]), dtype=float)
    exp_dirichlet_component = np.zeros((n_rtopics, all_exp_dirichlet_component.shape[1]), dtype=float)

    topic = 0
    for cluster in np.unique(clusters[f"leiden"]):
        tmp = clusters[clusters["leiden"] == cluster]

        tmp_components = all_components.loc[tmp.index, :]
        tmp_components = tmp_components.mean(axis=0)
        components[topic, :] = tmp_components.values

        tmp_exp_dirichlet_component = all_exp_dirichlet_component.loc[tmp.index, :]
        tmp_exp_dirichlet_component = tmp_exp_dirichlet_component.mean(axis=0)
        exp_dirichlet_component[topic, :] = tmp_exp_dirichlet_component.values

        topic = topic + 1

    df = pd.DataFrame(np.transpose(components),
                      columns=[f"Topic{i + 1}" for i in range(n_rtopics)],
                      index=all_components.columns)
    for topic in range(n_rtopics):
        df_sorted = df.sort_values([f"Topic{topic + 1}"], axis=0, ascending=False)[f"Topic{topic + 1}"]
        tmp = df_sorted.cumsum()
        tmp = tmp[tmp > 0.9 * df_sorted.sum()]
        df.loc[tmp.index, f"Topic{topic + 1}"] = 1 / n_rtopics

    components = df.T

    exp_dirichlet_component = pd.DataFrame(exp_dirichlet_component,
                                           index=[f"Topic{i + 1}" for i in range(n_rtopics)],
                                           columns=all_exp_dirichlet_component.columns)
    others = all_others.mean(axis=0)

    LDA = initialize_lda_model(components, exp_dirichlet_component, others)

    return LDA


def initialize_lda_model(components, exp_dirichlet_component, others):
    """
    Initialize LDA model by passing all necessary attributes
    :param components: Variational parameters for topic gene distribution
    :type components: pandas dataframe
    :param exp_dirichlet_component: Exponential value of expectation of log topic gene distribution
    :type exp_dirichlet_component: pandas dataframe
    :param others: dataframe contains remaining necessary attributes including
            n_batch_iter: Number of iterations of the EM step.
            n_features_in: Number of features seen during fit.
            n_iter: Number of passes over the dataset.
            bound: Final perplexity score on training set.
            doc_topic_prior: Prior of document topic distribution theta. If the value is None, it is 1 / n_components.
            topic_word_prior: Prior of topic word distribution beta. If the value is None, it is 1 / n_components.
    :type others: pandas dataframe
    :return: Latent Dirichlet Allocation with online variational Bayes algorithm.
    :rtype: sklearn.decomposition.LatentDirichletAllocation
    """
    n_topics = components.shape[0]

    LDA = LatentDirichletAllocation(n_components=n_topics)

    LDA.components_ = components.values
    LDA.exp_dirichlet_component_ = exp_dirichlet_component.values
    LDA.n_batch_iter_ = int(others['n_batch_iter'])
    LDA.n_features_in_ = int(others['n_features_in'])
    LDA.n_iter_ = int(others['n_iter'])
    LDA.bound_ = others['bound']
    LDA.doc_topic_prior_ = others['doc_topic_prior']
    LDA.topic_word_prior_ = others['topic_word_prior']

    return LDA


def filter_LDA_model(main_lda, keep):
    """
    filter LDA based on the topics we want to keep
    :param main_lda: Latent Dirichlet Allocation with online variational Bayes algorithm.
    :type main_lda: sklearn.decomposition.LatentDirichletAllocation
    :param keep: dataframe that define which topics we want to keep
    :type keep: pandas dataframe
    :return: Latent Dirichlet Allocation with online variational Bayes algorithm, weights of genes in each topics (indexes are topics and columns are genes)
    :rtype: sklearn.decomposition.LatentDirichletAllocation, pandas dataframe
    """
    n_topics = keep.sum()
    lda = LatentDirichletAllocation(n_components=n_topics)

    components = main_lda.components_[[i for i in range(keep.shape[0]) if keep[i]], :]

    df = pd.DataFrame(np.transpose(components),
                      columns=[f'Topic{i + 1}' for i in range(n_topics)])
    for topic in range(n_topics):
        df_sorted = df.sort_values([f'Topic{topic + 1}'], axis=0, ascending=False)[f'Topic{topic + 1}']
        tmp = df_sorted.cumsum()
        tmp = tmp[tmp > 0.9 * df_sorted.sum()]
        df.loc[tmp.index, f'Topic{topic + 1}'] = 1 / n_topics

    components = df.T

    lda.components_ = components.values
    lda.exp_dirichlet_component_ = main_lda.exp_dirichlet_component_[[i for i in range(keep.shape[0]) if keep[i]], :]
    lda.n_batch_iter_ = main_lda.n_batch_iter_
    lda.n_features_in_ = main_lda.n_features_in_
    lda.n_iter_ = main_lda.n_iter_
    lda.bound_ = main_lda.bound_
    lda.doc_topic_prior_ = main_lda.doc_topic_prior_
    lda.topic_word_prior_ = main_lda.topic_word_prior_

    return lda, components


def read_train(file):
    """
    reading train pickle file
    :param file: path of the pickle file
    :type file: str
    :return: train instance
    :rtype: Train class
    """
    if not os.path.isfile(file):
        raise ValueError('Train object not found at given path!')

    picklefile = open(file, 'rb')
    train = pickle.load(picklefile)

    print(f"Reading Train done!")
    return train


def read_topModel(file):
    """
    reading topModel pickle file
    :param file: path of the pickle file
    :type file: str
    :return: topModel instance
    :rtype: TopModel class
    """
    if not os.path.isfile(file):
        raise ValueError('TopModel object not found at given path!')

    picklefile = open(file, 'rb')
    topModel = pickle.load(picklefile)

    print(f"Reading TopModel done!")
    return topModel


def read_analysis(file):
    """
    reading analysis pickle file
    :param file: path of the pickle file
    :type file: str
    :return: analysis instance
    :rtype: Analysis class
    """
    if not os.path.isfile(file):
        raise ValueError('Analysis object not found at given path!')

    picklefile = open(file, 'rb')
    analysis = pickle.load(picklefile)

    print(f"Reading TopModel done!")
    return analysis
