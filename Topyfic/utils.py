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
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats as st
import scanpy.external as sce
import networkx as nx
import math

from Topyfic.train import *
from Topyfic.topic import *
from Topyfic.topModel import *
from Topyfic.analysis import *

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
    :param min_cell_participation: minimum cell participation across for each topics to keep them, when is None, it will keep topics with cell participation more than number of cell / 10000.
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
        min_cell_participation = data.to_df().shape[0] / 10000

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
        sce.pp.harmony_integrate(adata, 'assays')
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
                             rlda=rlda)

    else:
        name = np.unique(all_batches).tolist()
        name = '_'.join(name)

        top_model = TopModel(name=name,
                             N=n_rtopics,
                             gene_weights=gene_weights,
                             rlda=rlda)

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

    if not show_all:
        clustering = clustering[clustering["keep"]]

    options = np.unique(clustering[feature]).tolist()
    res = pd.DataFrame(columns=options,
                       index=range(len(clustering['leiden'].unique())))

    for i in range(res.shape[0]):
        for opt in options:
            tmp = clustering[np.logical_and(clustering['leiden'] == i,
                                            clustering[feature] == opt)]
            res.loc[i, opt] = tmp.shape[0]
    if not portion:
        plot = res.plot.bar(stacked=True,
                            xlabel='leiden',
                            ylabel='number of topics',
                            figsize=(max(res.shape[0] / 2, 5), 7))
    else:
        sum_feature = res.sum(axis=1)
        for i in range(res.shape[0]):
            for opt in options:
                res.loc[i, opt] = res.loc[i, opt] / sum_feature[i] * 100
        res.index = res.index + 1
        res.index = "Topics_" + res.index.astype(str)
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

    print(f"Reading Analysis done!")
    return analysis


def compare_topModels(topModels,
                      output_type='graph',
                      threshold=0.8,
                      topModels_color=None,
                      topModels_label=None,
                      save=False,
                      plot_show=True,
                      figsize=None,
                      plot_format="pdf",
                      file_name="compare_topics"):
    """
    compare several topModels

    :param topModels: list of topModel class you want to compare to each other
    :type topModels: list of TopModel class
    :param output_type: indicate the type of output you want. graph: plot as a graph, heatmap: plot as a heatmap, table: table contains correlation
    :type output_type: str
    :param threshold: only apply when you choose circular which only show correlation above that
    :type threshold: float
    :param topModels_color: dictionary of colors mapping each topics to each color (default: blue)
    :type topModels_color: dict
    :param topModels_label: dictionary of label mapping each topics to each label
    :type topModels_label: dict
    :param save: indicate if you want to save the plot or not (default: True)
    :type save: bool
    :param plot_show: indicate if you want to show the plot or not (default: True)
    :type plot_show: bool
    :param figsize: indicate the size of plot (default: (10 * (len(category) + 1), 10))
    :type figsize: tuple of int
    :param plot_format: indicate the format of plot (default: pdf)
    :type plot_format: str
    :param file_name: name and path of the plot use for save (default: piechart_topicAvgCell)
    :type file_name: str

    :return: table contains correlation between topics only when table is choose and save is False
    :rtype: pandas dataframe
    """
    if output_type not in ['graph', 'heatmap', 'table']:
        sys.exit("output_type is not valid! it should be one of 'graph', 'heatmap' or 'table'")

    all_gene_weights = None

    for topModel in topModels:
        gene_weights = topModel.get_gene_weights()
        if all_gene_weights is None:
            all_gene_weights = gene_weights
        else:
            all_gene_weights = pd.concat([gene_weights, all_gene_weights], axis=1)

    corrs = pd.DataFrame(index=all_gene_weights.columns,
                         columns=all_gene_weights.columns,
                         dtype='float64')

    for d1 in all_gene_weights.columns.tolist():
        for d2 in all_gene_weights.columns.tolist():
            if d1 == d2:
                corrs.at[d1, d2] = 1
                continue
            a = all_gene_weights[[d1, d2]]
            a.dropna(axis=0, how='all', inplace=True)
            a.fillna(0, inplace=True)
            corr = st.pearsonr(a[d1].tolist(), a[d2].tolist())
            corrs.at[d1, d2] = corr[0]

    if output_type == 'table':
        if save:
            corrs.to_csv(f"{file_name}.csv")
        else:
            return corrs

    if output_type == 'heatmap':
        sns.clustermap(corrs,
                       figsize=None)
        if save:
            plt.savefig(f"{file_name}.{plot_format}")
        if plot_show:
            plt.show()
        else:
            plt.close()

        return

    if output_type == 'graph':
        np.fill_diagonal(corrs.values, 0)
        corrs[corrs < threshold] = np.nan
        res = corrs.stack()
        res = pd.DataFrame(res)
        res.reset_index(inplace=True)
        res.columns = ['source', 'dest', 'weight']
        res['weight'] = res['weight'].astype(float).round(decimals=2)
        res['source_label'] = res['source']
        res['dest_label'] = res['dest']
        res['source_color'] = res['source']
        res['dest_color'] = res['dest']

        if topModels_label is not None:
            res['source_label'].replace(topModels_label, inplace=True)
            res['dest_label'].replace(topModels_label, inplace=True)
        if topModels_color is None:
            res['source_color'] = "blue"
            res['dest_color'] = "blue"
        else:
            res['source_color'].replace(topModels_color, inplace=True)
            res['dest_color'].replace(topModels_color, inplace=True)

        G = nx.Graph()
        for i in range(res.shape[0]):
            G.add_node(res.source_label[i], color=res.source_color[i])
            G.add_node(res.dest_label[i], color=res.dest_color[i])
            G.add_edge(res.source_label[i], res.dest_label[i], weight=res.weight[i])

        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

        if figsize is None:
            figsize = (len(connected_components) * 2, len(connected_components) * 2)

        nrows = math.ceil(math.sqrt(len(connected_components)))
        ncols = math.ceil(len(connected_components) / nrows)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor='white')

        i = 0
        for connected_component in connected_components:
            g_connected_component = G.subgraph(connected_component)
            nodePos = nx.spring_layout(g_connected_component)

            edge_labels = nx.get_edge_attributes(g_connected_component, "weight")

            node_color = nx.get_node_attributes(g_connected_component, "color").values()
            weights = nx.get_edge_attributes(g_connected_component, 'weight').values()

            if len(connected_components) == 1:
                ax = axs
            else:
                ax = axs[int(i / ncols), i % ncols]

            nx.draw_networkx(g_connected_component,
                             pos=nodePos,
                             width=list(weights),
                             with_labels=True,
                             node_color=list(node_color),
                             font_size=8,
                             node_size=500,
                             ax=ax)

            nx.draw_networkx_edge_labels(g_connected_component,
                                         nodePos,
                                         edge_labels=edge_labels,
                                         font_size=7,
                                         ax=ax)

            i += 1

        [axi.axis('off') for axi in axs.ravel()]
        plt.tight_layout()
        if save:
            plt.savefig(f"{file_name}.{plot_format}")
        if plot_show:
            plt.show()
        else:
            plt.close()

        return axs


def MA_plot(topic1,
                topic2,
                pseudocount=1,
                threshold=1,
                cutoff=2,
                consistency_correction=1.4826,
                labels=None,
                save=True,
                show=True,
                file_format="pdf",
                file_name="MA_plot"):
        """
        plot MA based on the gene weights on given topics

        :param topic1: first topic to be compared
        :type topic1: str
        :param topic2: second topic to be compared
        :type topic2: str
        :param pseudocount: pseudocount that you want to add (default: 1)
        :type pseudocount: float
        :param threshold: threshold to filter genes based on A values (default: 1)
        :type threshold: float
        :param cutoff: cutoff for categorized genes by modified z-score (default: 2)
        :type cutoff: float
        :param consistency_correction: the factor converts the MAD to the standard deviation for a given distribution. The default value (1.4826) is the conversion factor if the underlying data is normally distributed
        :type consistency_correction: float
        :param labels: list of gene names wish to show in MA-plot
        :type labels: list
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: MA_plot)
        :type file_name: str

        :return: return M and A values
        """
        topic1 += pseudocount
        topic2 += pseudocount

        A = (np.log2(topic1) + np.log2(topic2)) / 2
        M = np.log2(topic1) - np.log2(topic2)

        gene_zscore = pd.concat([A, M], axis=1)
        gene_zscore.columns = ["A", "M"]
        gene_zscore = gene_zscore[gene_zscore.A > threshold]

        gene_zscore['mod_zscore'], mad = modified_zscore(gene_zscore['M'],
                                                         consistency_correction=consistency_correction)

        plot_df = gene_zscore.copy(deep=True)
        plot_df.mod_zscore = plot_df.mod_zscore.abs()
        plot_df.mod_zscore[plot_df.mod_zscore > cutoff] = cutoff
        plot_df.mod_zscore[plot_df.mod_zscore < cutoff] = 0
        plot_df.mod_zscore.fillna(0, inplace=True)
        plot_df.mod_zscore = plot_df.mod_zscore.astype(int)
        if labels is not None:
            plot_df['label'] = plot_df.index.tolist()
            plot_df.label[~plot_df.label.isin(labels)] = ""

        y = plot_df.M.median()
        ymin = y - consistency_correction * mad * 2
        ymax = y + consistency_correction * mad * 2
        xmin = round(gene_zscore.A.min()) - 1
        xmax = round(gene_zscore.A.max())

        sns.scatterplot(data=plot_df, x="A", y="M", hue="mod_zscore",
                        palette=["orchid", "royalblue"])

        if labels is not None:
            for label in labels:
                plt.text(plot_df.A[label] - 0.2, plot_df.M[label] + 0.2, label)

        plt.legend(title='abs(Z-score)', loc='upper right', labels=[f'> {cutoff}', f'< {cutoff}'])

        plt.hlines(y=y, xmin=xmin, xmax=xmax, colors="red")
        plt.hlines(y=ymin, xmin=xmin, xmax=xmax, colors="orange", linestyles='--')
        plt.hlines(y=ymax, xmin=xmin, xmax=xmax, colors="orange", linestyles='--')

        if save:
            plt.savefig(f"{file_name}.{file_format}")
        if show:
            plt.show()
        else:
            plt.close()

        return gene_zscore


def modified_zscore(data, consistency_correction=1.4826):
    """
    Returns the modified z score and Median Absolute Deviation (MAD) from the scores in data.
    The consistency_correction factor converts the MAD to the standard deviation for a given
    distribution. The default value (1.4826) is the conversion factor if the underlying data
    is normally distributed
    """

    median = np.median(data)

    deviation_from_med = np.array(data) - median

    mad = np.median(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med / (consistency_correction * mad)

    return mod_zscore, mad


def functional_enrichment_analysis(gene_list,
                                       type,
                                       organism,
                                       sets=None,
                                       p_value=0.05,
                                       file_format="pdf",
                                       file_name="functional_enrichment_analysis"):
    """
    Doing functional enrichment analysis including GO, KEGG and REACTOME

    :param gene_list: list of gene name
    :type gene_list:list
    :param type: indicate the type of databases which it should be one of "GO", "REACTOME"
    :type type: str
    :param organism: name of the organ you want to do functional enrichment analysis
    :type organism: str
    :param sets: str, list, tuple of Enrichr Library name(s). (you can add any Enrichr Libraries from here: https://maayanlab.cloud/Enrichr/#stats) only need to fill if the type is GO
    :type sets: str, list, tuple
    :param p_value: Defines the pValue threshold. (default: 0.05)
    :type p_value: float
    :param file_format: indicate the format of plot (default: pdf)
    :type file_format: str
    :param file_name: name and path of the plot use for save (default: gene_composition)
    :type file_name: str
    """

    if type not in ["GO", "REACTOME"]:
        sys.exit("Type is not valid! it should be one of them GO, KEGG, REACTOME")

    if type == "GO" and sets is None:
        sets = ["GO_Biological_Process_2021"]
    elif type == "KEGG" and sets is None:
        sets = ["KEGG_2016"]

    if type in ["GO", "KEGG"]:
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=sets,
                         organism=organism,
                         outdir=f"{file_name}",
                         cutoff=p_value)
        dotplot(enr.res2d,
                title=f"Gene ontology",
                cmap='viridis_r',
                cutoff=p_value,
                ofname=f"{file_name}.{file_format}")
    else:
        numGeneModule = len(gene_list)
        genes = ",".join(gene_list)
        result = analysis.identifiers(ids=genes,
                                      species=organism,
                                      p_value=str(p_value))
        token = result['summary']['token']
        analysis.report(token,
                        path=f"{file_name}/",
                        file=f"{file_name}.{file_format}",
                        number='50',
                        species=organism)
        token_result = analysis.token(token,
                                      species=organism,
                                      p_value=str(p_value))

        print(
            f"{numGeneModule - token_result['identifiersNotFound']} out of {numGeneModule} identifiers in the sample were found in Reactome.")
        print(
            f"{token_result['resourceSummary'][0]['pathways']} pathways were hit by at least one of them, which {len(token_result['pathways'])} of them have p-value more than {p_value}.")
        print(f"Report was saved {file_name}!")
        print(f"For more information please visit https://reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS={token}")


def GSEA(gene_list,
             gene_sets='GO_Biological_Process_2021',
             p_value=0.05,
             table=True,
             plot=True,
             file_format="pdf",
             file_name="GSEA",
             **kwargs):
    """
    Doing Gene Set Enrichment Analysis on based on the topic weights using GSEAPY package.

    :param gene_list: pandas series with index as a gene names and their ranks/weights as value
    :type gene_list: pandas series
    :param gene_sets: Enrichr Library name or .gmt gene sets file or dict of gene sets. (you can add any Enrichr Libraries from here: https://maayanlab.cloud/Enrichr/#stats)
    :type gene_sets: str, list, tuple
    :param p_value: Defines the pValue threshold for plotting. (default: 0.05)
    :type p_value: float
    :param table: indicate if you want to save all GO terms that passed the threshold as a table (default: True)
    :type table: bool
    :param plot: indicate if you want to plot all GO terms that passed the threshold (default: True)
    :type plot: bool
    :param file_format: indicate the format of plot (default: pdf)
    :type file_format: str
    :param file_name: name and path of the plot use for save (default: gene_composition)
    :type file_name: str
    :param kwargs: Argument to pass to gseapy.prerank(). more info: https://gseapy.readthedocs.io/en/latest/run.html?highlight=gp.prerank#gseapy.prerank

    :return: dataframe contains these columns: Term: gene set name, ES: enrichment score, NES: normalized enrichment score, NOM p-val:  Nominal p-value (from the null distribution of the gene set, FDR q-val: FDR qvalue (adjusted False Discory Rate), FWER p-val: Family wise error rate p-values, Tag %: Percent of gene set before running enrichment peak (ES), Gene %: Percent of gene list before running enrichment peak (ES), Lead_genes: leading edge genes (gene hits before running enrichment peak)
    :rtype: pandas dataframe
    """

    gene_list.index = gene_list.index.str.upper()

    pre_res = gp.prerank(rnk=gene_list,
                         gene_sets=gene_sets,
                         format=file_format,
                         no_plot=~plot,
                         **kwargs)

    pre_res.res2d.sort_values(["NOM p-val"], inplace=True)
    pre_res.res2d.drop(["Name"], axis=1, inplace=True)

    if table:
        pre_res.res2d.to_csv(f"{file_name}.csv")

    if plot:
        res = pre_res.res2d.copy(deep=True)
        res = res[res['NOM p-val'] < p_value]
        for term in res.Term:
            name = term.split("(GO:")[1][:-1]
            gseaplot(rank_metric=pre_res.ranking,
                     term=term,
                     **pre_res.results[term],
                     ofname=f"{file_name}_GO_{name}.{file_format}")

    return pre_res.res2d
