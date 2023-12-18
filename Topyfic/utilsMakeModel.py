import sys
import random
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
from sklearn.decomposition import LatentDirichletAllocation, NMF
from scipy import stats as st
import scanpy.external as sce
import networkx as nx
import math
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from gseapy.plot import dotplot
from gseapy import gseaplot
from reactome2py import analysis
import yaml
from yaml.loader import SafeLoader
import h5py

from Topyfic.train import Train
from Topyfic.analysis import Analysis
from Topyfic.topModel import TopModel
from Topyfic.topic import Topic

warnings.filterwarnings("ignore")


def train_model(name,
                data,
                k,
                n_runs=100,
                random_state_range=None,
                n_thread=5,
                save_path=""):
    """
    Training model and save it

    :param name: name of the Train class
    :type name: str
    :param k: number of topics to learn one LDA model using sklearn package (default: 50)
    :type k: int
    :param n_runs: number of run to define rLDA model (default: 100)
    :type n_runs: int
    :param random_state_range: list of random state, we used to run LDA models (default: range(n_runs))
    :type random_state_range: list of int
    :param data: data embedded in anndata format use to train LDA model
    :type data: anndata
    :param n_thread: number of threads you used to learn LDA models (default=5)
    :type n_thread: int
    :param save_path: directory you want to use to save pickle file (default is saving near script)
    :type save_path: str

    """
    train = Train(name=name,
                  k=k,
                  n_runs=n_runs,
                  random_state_range=random_state_range)
    train.run_LDA_models(data, n_thread=n_thread)
    train.save_train(save_path=save_path)


def make_topModel(trains,
                  data,
                  n_top_genes=50,
                  resolution=1,
                  file_format="pdf",
                  save_path=""):
    """
    Creating topModel base on train data and save it along with clustering information

    :param trains: list of train class
    :type trains: list of Train
    :param data: expression data embedded in anndata format along with cells and genes/region information
    :type data: anndata
    :param n_top_genes: Number of highly-variable genes to keep (default: 50)
    :type n_top_genes: int
    :param resolution: A parameter value controlling the coarseness of the clustering. Higher values lead to more clusters. (default: 1)
    :type resolution: int
    :param file_format: indicate the format of plot (default: pdf)
    :type file_format: str
    :param save_path: directory you want to use to save pickle file (default is saving near script)
    :type save_path: str

    """
    top_model, clustering = calculate_leiden_clustering(trains=trains,
                                                        data=data,
                                                        n_top_genes=n_top_genes,
                                                        resolution=resolution,
                                                        file_format=file_format)
    clustering.to_csv(f"{save_path}/clustering.csv")
    top_model.save_topModel(save_path=save_path)


def make_analysis_class(top_model,
                        data,
                        colors_topics=None,
                        save_path=""):
    """
    Creating Analysis object

    :param top_model: top model that used for analysing topics, gene weights compositions and calculate cell participation
    :type top_model: TopModel
    :param data: processed expression data along with cells and genes/region information
    :type data: anndata
    :param colors_topics: dataframe that mapped colored to topics
    :type colors_topics: pandas dataframe
    :param save_path: directory you want to use to save pickle file (default is saving near script)
    :type save_path: str
    """
    analysis_top_model = Analysis(Top_model=top_model, colors_topics=colors_topics)
    analysis_top_model.calculate_cell_participation(data=data)
    analysis_top_model.save_analysis(save_path=save_path)


def subset_data(data, keep, loc='var'):
    """
    Subsetting data

    :param data: data we want to subset
    :type data: anndata
    :param keep: values in the obs/var_names
    :type keep: list
    :param loc: subsetting in which direction (default: 'var')

    :return: data we want to keep
    :rtype: anndata
    """
    if loc == 'var':
        data = data[:, keep]
    elif loc == 'obs':
        data = data[keep, :]
    else:
        sys.exit("loc is not correct! it should be 'obs' or 'var'")

    return data


def calculate_leiden_clustering(trains,
                                data,
                                n_top_genes=50,
                                resolution=1,
                                max_iter_harmony=10,
                                min_cell_participation=None,
                                file_format="pdf"):
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
    :param max_iter_harmony: number of iteration for running harmony (default: 10)
    :type max_iter_harmony: int
    :param min_cell_participation: minimum cell participation across for each topics to keep them, when is None, it will keep topics with cell participation more than 1% of #cells (#cells / 100)
    :type min_cell_participation: float
    :param file_format: indicate the format of plot (default: pdf)
    :type file_format: str

    :return: final TopModel instance after clustering and trimming, dataframe containing which run goes to which topic
    :rtype: TopModel, pandas dataframe
    """
    all_batches = None
    all_components = None
    all_exp_dirichlet_component = None
    all_others = None

    if min_cell_participation is None:
        min_cell_participation = data.shape[0] / 100

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
            all_components = pd.concat([components, all_components], axis=0)
            all_exp_dirichlet_component = pd.concat([exp_dirichlet_component, all_exp_dirichlet_component], axis=0)
            all_others = pd.concat([others, all_others], axis=0)

    if len(trains) == 1:
        adata = anndata.AnnData(all_components)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
        adata.obs["topics"] = adata.obs["leiden"].astype(int) + 1
        adata.obs["topics"] = "Topic_" + adata.obs["topics"].astype(str)
        sc.pl.umap(adata, color=["topics"],
                   title=[f"Topic space UMAP leiden clusters (k={trains[0].k})"],
                   save=f"_leiden_clustering.{file_format}")
        sc.pl.umap(adata, color=['topics'],
                   legend_loc='on data',
                   title=[f'Topic space UMAP leiden clusters'],
                   save=f'_leiden_clustering_v2.{file_format}')

    else:
        adata = anndata.AnnData(all_components)
        adata.obs['assays'] = all_batches
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        sc.pp.neighbors(adata)
        sce.pp.harmony_integrate(adata, 'assays', max_iter_harmony=max_iter_harmony)
        adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
        adata.obs["topics"] = adata.obs["leiden"].astype(int) + 1
        adata.obs["topics"] = "Topic_" + adata.obs["topics"].astype(str)
        sc.pl.umap(adata, color=['topics'],
                   title=[f'Topic space UMAP leiden clusters'],
                   save=f'_leiden_clustering_harmony.{file_format}')
        sc.pl.umap(adata, color=['topics'],
                   legend_loc='on data',
                   title=[f'Topic space UMAP leiden clusters'],
                   save=f'_leiden_clustering_harmony_v2.{file_format}')
        sc.pl.umap(adata, color=['assays'],
                   title=['Topic space UMAP leiden clusters'],
                   save=f'_technology_harmony.{file_format}')

    clustering = adata.obs.copy(deep=True)
    clustering["leiden"] = clustering["leiden"].astype(int)
    n_rtopics = len(np.unique(clustering["leiden"]))

    rlda = initialize_rLDA_model(all_components,
                                 all_exp_dirichlet_component,
                                 all_others,
                                 clusters=clustering)
    lda_output = rlda.transform(data.X)
    cell_participation = pd.DataFrame(np.round(lda_output, 2),
                                      columns=[f"Topic{i + 1}" for i in range(n_rtopics)],
                                      index=data.obs.index)

    keep = cell_participation.sum() > min_cell_participation
    print(
        f"{keep.sum()} topics out of {keep.shape[0]} topics have participation more than {min_cell_participation}")
    tmp = pd.DataFrame(keep)
    tmp["cluster"] = range(tmp.shape[0])
    tmp = tmp.to_dict()
    tmp = tmp[0]
    clustering["keep"] = clustering["leiden"].astype(int) + 1
    clustering["keep"] = "Topic" + clustering["keep"].astype(str)
    clustering["keep"] = clustering["keep"].replace(tmp)
    n_rtopics = keep.sum()
    rlda, gene_weights_T = filter_LDA_model(rlda, keep)

    gene_weights = gene_weights_T.T
    gene_weights.index = data.var.index

    if len(trains) == 1:
        top_model = TopModel(name=trains[0].name,
                             N=n_rtopics,
                             gene_weights=gene_weights,
                             model=rlda)

    else:
        name = np.unique(all_batches).tolist()
        name = '_'.join(name)

        top_model = TopModel(name=name,
                             N=n_rtopics,
                             gene_weights=gene_weights,
                             model=rlda)

    return top_model, clustering, adata


def plot_cluster_contribution(clustering,
                              feature,
                              show_all=False,
                              portion=True,
                              save=True,
                              show=True,
                              file_format="pdf",
                              file_name='cluster_contribution'):
    """
    barplot shows number of topics contribute to each cluster

    :param clustering: dataframe that map each single LDA run to each cluster
    :type clustering: pandas dataframe
    :param feature: name of the feature you want to see the cluster contribution (should be one of the columns name of clustering df)
    :type feature: str
    :param show_all: Indicate if you want to show all clusters or only the ones that pass threshold (default: False)
    :type show_all: bool
    :param portion: Indicate if you want to normalized the bar to show percentage instead of actual value (default: True)
    :type portion: bool
    :param save: indicate if you want to save the plot or not (default: True)
    :type save: bool
    :param show: indicate if you want to show the plot or not (default: True)
    :type show: bool
    :param file_format: indicate the format of plot (default: pdf)
    :type file_format: str
    :param file_name: name and path of the plot use for save (default: cluster_contribution)
    :type file_name: str
    """
    if feature not in clustering.columns:
        sys.exit(f"{feature} is not valid! should be a columns names of clustering")

    options = np.unique(clustering['assays']).tolist()
    res = pd.DataFrame(columns=options + ['Topic', 'keep'],
                       index=range(len(clustering['leiden'].unique())))

    for i in clustering['leiden'].unique():
        for opt in options:
            tmp = clustering[np.logical_and(clustering['leiden'] == i,
                                            clustering['assays'] == opt)]
            res.loc[i, opt] = tmp.shape[0]

        tmp = clustering[clustering['leiden'] == i]
        res.loc[i, 'Topic'] = f"Topic_{i + 1}"
        res.loc[i, 'keep'] = tmp.keep.unique()[0]

    if not show_all:
        res = res[res["keep"]]
    res.index = res.Topic.values

    if not portion:
        plot = res.plot.bar(stacked=True,
                            xlabel='leiden',
                            ylabel='number of topics',
                            figsize=(max(res.shape[0] / 2, 5), 7))
    else:
        res['sum'] = res[options].sum(axis=1)
        for i in res.index.values:
            for opt in options:
                res.loc[i, opt] = float(res.loc[i, opt]) / res.loc[i, 'sum'] * 100

        res.drop(['Topic', 'keep', 'sum'], axis=1, inplace=True)

        plot = res.plot.bar(stacked=True,
                            xlabel='leiden',
                            ylabel='percentage(%) of topics',
                            figsize=(max(res.shape[0] / 2, 5), 7))
    fig = plot.get_figure()
    if save:
        fig.savefig(f"{file_name}.{file_format}", bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def initialize_rLDA_model(all_components, all_exp_dirichlet_component, all_others, clusters):
    """
    Initialize reproducible LDA model by calculating all necessary attributes using clustering.

    :param all_components: Variational parameters for topic gene distribution from all single LDA models
    :type all_components: pandas dataframe
    :param all_exp_dirichlet_component: Exponential value of expectation of log topic gene distribution from all single LDA models
    :type all_exp_dirichlet_component: pandas dataframe
    :param all_others: dataframe contains remaining necessary attributes including: n_batch_iter: Number of iterations of the EM step. n_features_in: Number of features seen during fit. n_iter: Number of passes over the dataset. bound: Final perplexity score on training set. doc_topic_prior: Prior of document topic distribution theta. If the value is None, it is 1 / n_components. topic_word_prior: Prior of topic word distribution beta. If the value is None, it is 1 / n_components.
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
    for cluster in range(n_rtopics):
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
    :param others: dataframe contains remaining necessary attributes including: n_batch_iter: Number of iterations of the EM step. n_features_in: Number of features seen during fit. n_iter: Number of passes over the dataset. bound: Final perplexity score on training set. doc_topic_prior: Prior of document topic distribution theta. If the value is None, it is 1 / n_components. topic_word_prior: Prior of topic word distribution beta. If the value is None, it is 1 / n_components.
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
        raise ValueError('Train file not found at given path!')
    if not file.endswith('.p') and not file.endswith('.h5'):
        raise ValueError('Train file type is not correct!')

    if file.endswith('.p'):
        picklefile = open(file, 'rb')
        train = pickle.load(picklefile)

    if file.endswith('.h5'):
        f = h5py.File(file, 'r')

        name = np.string_(f['name']).decode('ascii')
        k = np.int_(f['k'])
        n_runs = np.int_(f['n_runs'])
        random_state_range = list(f['random_state_range'])

        # models
        top_models = []
        for random_state in random_state_range:
            components = pd.DataFrame(np.array(f[f"models/{random_state}/components_"]))
            exp_dirichlet_component = pd.DataFrame(np.array(f[f"models/{random_state}/exp_dirichlet_component_"]))

            others = pd.DataFrame()
            others.loc[0, 'n_batch_iter'] = np.int_(f[f"models/{random_state}/n_batch_iter_"])
            others.loc[0, 'n_features_in'] = np.array(f[f"models/{random_state}/n_features_in_"])
            others.loc[0, 'n_iter'] = np.int_(f[f"models/{random_state}/n_iter_"])
            others.loc[0, 'bound'] = np.float_(f[f"models/{random_state}/bound_"])
            others.loc[0, 'doc_topic_prior'] = np.array(f[f"models/{random_state}/doc_topic_prior_"])
            others.loc[0, 'topic_word_prior'] = np.array(f[f"models/{random_state}/topic_word_prior_"])

            model = initialize_lda_model(components, exp_dirichlet_component, others)

            top_model = TopModel(name=f"{name}_{random_state}",
                                 N=k,
                                 gene_weights=components,
                                 model=model)
            top_models.append(top_model)

        train = Train(name=name,
                      k=k,
                      n_runs=n_runs,
                      random_state_range=random_state_range)
        train.top_models = top_models

        f.close()

    print(f"Reading Train done!")
    return train


def read_topModel(file):
    """
    reading topModel pickle/HDF5 file

    :param file: path of the pickle/HDF5 file
    :type file: str

    :return: topModel instance
    :rtype: TopModel class
    """
    if not os.path.isfile(file):
        raise ValueError('TopModel file not found at given path!')
    if not file.endswith('.p') and not file.endswith('.h5'):
        raise ValueError('TopModel file type is not correct!')

    if file.endswith('.p'):
        picklefile = open(file, 'rb')
        top_model = pickle.load(picklefile)

    if file.endswith('.h5'):
        f = h5py.File(file, 'r')

        name = np.string_(f['name']).decode('ascii')
        N = np.int_(f['N'])

        # topics
        topics = dict()
        topic_ids = [f'Topic_{i + 1}' for i in range(N)]
        for topic in topic_ids:
            topic_id = np.string_(f['topics'][topic]['id']).decode('ascii')
            topic_name = np.string_(f['topics'][topic]['name']).decode('ascii')
            gene_weights = pd.DataFrame(np.array(f['topics'][topic]['gene_weights']))
            gene_information = pd.DataFrame(np.array(f['topics'][topic]['gene_information']), dtype=str)
            gene_information.columns = gene_information.iloc[0, :]
            gene_information.drop(index=0, inplace=True)
            gene_information.index = gene_information['index']
            gene_information.drop(columns='index', inplace=True)
            gene_information.index.name = None

            topic_information = pd.DataFrame(np.array(f['topics'][topic]['topic_information']), dtype=str)
            topic_information.columns = topic_information.iloc[0, :]
            topic_information.drop(index=0, inplace=True)
            topic_information.index = topic_information['index']
            topic_information.drop(columns='index', inplace=True)
            topic_information.index.name = None

            gene_weights.index = gene_information.index.tolist()
            gene_weights.columns = topic_information.index.tolist()

            topic = Topic(topic_id=topic_id,
                          topic_name=topic_name,
                          topic_gene_weights=gene_weights,
                          gene_information=gene_information,
                          topic_information=topic_information)
            topics[topic_id] = topic

        # model
        components = pd.DataFrame(np.array(f['model']['components_']))
        exp_dirichlet_component = pd.DataFrame(np.array(f['model']['exp_dirichlet_component_']))

        others = pd.DataFrame()
        others.loc[0, 'n_batch_iter'] = np.int_(f['model']['n_batch_iter_'])
        others.loc[0, 'n_features_in'] = np.array(f['model']['n_features_in_'])
        others.loc[0, 'n_iter'] = np.int_(f['model']['n_iter_'])
        others.loc[0, 'bound'] = np.float_(f['model']['bound_'])
        others.loc[0, 'doc_topic_prior'] = np.array(f['model']['doc_topic_prior_'])
        others.loc[0, 'topic_word_prior'] = np.array(f['model']['topic_word_prior_'])

        model = initialize_lda_model(components, exp_dirichlet_component, others)

        top_model = TopModel(name=name,
                             N=N,
                             topics=topics,
                             model=model)

        f.close()

    print(f"Reading TopModel done!")
    return top_model


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

    print(f"Reading Analysis done!")
    return analysis


def read_model_yaml(model_yaml_path="model.yaml",
                    topic_yaml_path=None,
                    cell_topic_participation_path=None,
                    save=True):
    """
    read YMAL files and make topmodel object
    write topic in YAML format

    :param model_yaml_path: model yaml path
    :type model_yaml_path: str
    :param topic_yaml_path: path that you use to save all topics information
    :type topic_yaml_path: str
    :param cell_topic_participation_path: path of cell-topic participation
    :type cell_topic_participation_path: str
    :param save: indicate if you want to save objects (topmodel and analysis) as a pickle file (default: True)
    :type save: bool

    :return: Topmodel and analysis objects
    :rtype: TopModel, Analysis
    """

    with open(model_yaml_path, 'r') as file:
        model_yaml = yaml.safe_load(file)

    if not all(value in list(model_yaml.keys()) for value in
               ['Topic IDs', 'Cell-Topic participation ID', 'Experiment ID', 'Name of method', 'Number of topics']):
        sys.exit("Model YMAL file is not correct!")

    topics = dict()
    topics_gene_weights = None
    for i in range(len(model_yaml['Topic IDs'])):
        topic_path = f"{topic_yaml_path}{model_yaml['Topic IDs'][i]}.yaml"
        with open(topic_path, 'r') as file:
            topic_yaml = yaml.safe_load(file)

        if not all(value in list(topic_yaml.keys()) for value in
                   ['Topic ID', 'Gene weights', 'Gene information']):
            sys.exit(f"Topic {model_yaml['Topic IDs'][i]} YMAL file is not correct!")

        topic_id = topic_yaml['Topic ID']
        topic_gene_weights = pd.DataFrame(list(topic_yaml['Gene weights'].values()),
                                          index=topic_yaml['Gene weights'].keys(),
                                          columns=[topic_yaml['Topic ID']])
        mask = topic_gene_weights.applymap(lambda x: isinstance(x, (int, float)))

        topic_gene_weights = topic_gene_weights.where(mask)
        gene_information = pd.DataFrame(topic_yaml['Gene information'])
        topic_information = None
        if 'Topic information' in topic_yaml.keys():
            topic_information = pd.DataFrame(topic_yaml['Topic information'], index=topic_gene_weights.columns)
        topics[f"Topic_{i + 1}"] = Topic(topic_id=topic_id,
                                         topic_gene_weights=topic_gene_weights,
                                         gene_information=gene_information,
                                         topic_information=topic_information)
        if topics_gene_weights is None:
            topics_gene_weights = topic_gene_weights.copy(deep=True)
        else:
            topics_gene_weights = pd.concat([topics_gene_weights, topic_gene_weights], axis=1)

    if "NMF" in model_yaml['Name of method']:
        model = NMF(n_components=int(model_yaml['Number of topics']))
    else:
        model = LatentDirichletAllocation(n_components=int(model_yaml['Number of topics']))

    model.components_ = topics_gene_weights.T.values
    topModel = TopModel(name=model_yaml['Experiment ID'],
                        N=int(model_yaml['Number of topics']),
                        topics=topics,
                        model=model)

    cell_participation = sc.read_h5ad(cell_topic_participation_path)
    analysis = Analysis(Top_model=topModel,
                        cell_participation=cell_participation)

    if save:
        topModel.save_topModel()
        analysis.save_analysis()

    return topModel, analysis


def combine_topModels(topModels,
                      name="Combined_TopModel",
                      data=None,
                      min_cell_participation=None):
    """
    Combine two topmodels. It will not apply any method when we want to combine them, so basically just combine all models without performing any method

    :param topModels: list of topmodels you want to combine
    :type topModels: list of TopModel
    :param name: name of the combined topmodels
    :type name: str
    :param data: if you want to remove topics with low cell participation, you can pass the data you used to train models
    :type data: anndata
    :param min_cell_participation: minimum cell participation across for each topics to keep them, when is None, it will keep topics with cell participation more than 1% of #cells (#cells / 100)
    :type min_cell_participation: float

    :return: return the combined TopModel, number of topics, gene weights
    :rtype: TopModel, int, pandas DataFrame
    """
    n_topics = 0
    components = None
    exp_dirichlet_component = None
    n_batch_iter = 0
    n_features_in = 0
    n_iter = 0
    bound = 0.0
    doc_topic_prior = 0.0
    topic_word_prior = 0.0
    for topmodel in topModels:
        n_topics += topmodel.N

        tmp = pd.DataFrame(topmodel.model.components_,
                           index=[f'{topmodel.name}_Topic_{i + 1}' for i in
                                  range(topmodel.model.components_.shape[0])],
                           columns=topmodel.get_feature_name())
        if components is None:
            components = tmp
        else:
            components = pd.concat([components, tmp], axis=0)

        tmp = pd.DataFrame(topmodel.model.exp_dirichlet_component_,
                           index=[f'{topmodel.name}_Topic_{i + 1}' for i in
                                  range(topmodel.model.components_.shape[0])],
                           columns=topmodel.get_feature_name())
        if exp_dirichlet_component is None:
            exp_dirichlet_component = tmp
        else:
            exp_dirichlet_component = pd.concat([exp_dirichlet_component, tmp], axis=0)

        n_batch_iter += topmodel.N * topmodel.model.n_batch_iter_
        n_features_in += topmodel.N * topmodel.model.n_features_in_
        n_iter += topmodel.N * topmodel.model.n_iter_
        bound += topmodel.N * topmodel.model.bound_
        doc_topic_prior += topmodel.N * topmodel.model.doc_topic_prior_
        topic_word_prior += topmodel.N * topmodel.model.topic_word_prior_

    n_batch_iter /= n_topics
    n_features_in /= n_topics
    n_iter /= n_topics
    bound /= n_topics
    doc_topic_prior /= n_topics
    topic_word_prior /= n_topics

    model = LatentDirichletAllocation(n_components=n_topics)

    model.components_ = components.values
    model.exp_dirichlet_component_ = exp_dirichlet_component.values
    model.n_batch_iter_ = int(n_batch_iter)
    model.n_features_in_ = int(n_features_in)
    model.n_iter_ = int(n_iter)
    model.bound_ = bound
    model.doc_topic_prior_ = doc_topic_prior
    model.topic_word_prior_ = topic_word_prior

    if data is not None:
        if min_cell_participation is None:
            min_cell_participation = data.shape[0] / 100

        lda_output = model.transform(data.X)
        cell_participation = pd.DataFrame(np.round(lda_output, 2),
                                          columns=[f"Topic{i + 1}" for i in range(n_topics)],
                                          index=data.obs.index)

        keep = cell_participation.sum() > min_cell_participation
        print(
            f"{keep.sum()} topics out of {keep.shape[0]} topics have participation more than {min_cell_participation}")
        n_topics = keep.sum()
        model, tmp = filter_LDA_model(model, keep)
        components = pd.DataFrame(model.components_,
                                  index=[f'{name}_Topic_{i + 1}' for i in
                                         range(model.components_.shape[0])],
                                  columns=components.columns.tolist())

    top_model = TopModel(name=name,
                         N=n_topics,
                         gene_weights=components.T,
                         model=model)

    return top_model, n_topics, components.T
