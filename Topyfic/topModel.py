import sys
import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

sns.set_context('paper')
warnings.filterwarnings('ignore')

from Topyfic.topic import Topic
from Topyfic.utilsAnalyseModel import MA_plot


class TopModel:
    """
    A class that saved a model

    :param name: name of class
    :type name: str
    :param N: number of topics
    :type N: int
    :param gene_weights: dataframe that has weights of genes for each topics; genes are indexes and topics are columns
    :type gene_weights: pandas dataframe
    :param topics: dictionary contains all topics for the topmodel
    :type topics: Dictionary of Topics
    :param model: store reproducible LDA model
    :type model: sklearn.decomposition.LatentDirichletAllocation

    """

    def __init__(self,
                 name,
                 N,
                 topics=None,
                 gene_weights=None,
                 gene_information=None,
                 model=None):
        if gene_weights is None and topics is None:
            sys.exit("Both gene weights and topics can not be empty at the same time!")

        if topics is not None:
            self.topics = topics
        else:
            self.topics = dict()
            for i in range(N):
                topic_gene_weights = pd.DataFrame(gene_weights.iloc[:, i].values,
                                                  index=gene_weights.index,
                                                  columns=[f"Topic_{i + 1}"])
                self.topics[f"Topic_{i + 1}"] = Topic(topic_id=f"Topic_{i + 1}",
                                                      topic_gene_weights=topic_gene_weights,
                                                      gene_information=gene_information)
        self.name = name
        self.N = N
        self.model = model

    def get_feature_name(self):
        """
        get feature(gene) name

        :return: list of feature(gene) name
        :rtype: list
        """
        topic1 = next(iter(self.topics.items()))[1]
        return topic1.gene_information.index.tolist()

    def save_rLDA_model(self, name='rLDA', save_path="", file_format='joblib'):
        """
            save rLDA model (instance of LDA model in sklearn) as a joblib/HDF5 file.

            :param name: name of the joblib file (default: rLDA)
            :type name: str
            :param save_path: directory you want to use to save pickle file (default is saving near script)
            :type save_path: str
            :param file_format: format of the file you want to save (option: joblib (default), HDF5)
            :type file_format: str
        """
        if file_format not in ['joblib', 'HDF5']:
            sys.exit(f"{file_format} is not correct! It should be 'joblib' or 'HDF5'.")

        if file_format == "joblib":
            print(f"Saving rLDA model as {name}_{self.N}topics.joblib")

            joblib.dump(self.model, f"{save_path}{name}_{self.N}topics.joblib", compress=3)

        if file_format == "HDF5":
            print(f"Saving rLDA model as {name}_{self.N}topics.h5")

            f = h5py.File(f"{name}_{self.N}topics.h5", "a")

            f['components_'] = self.model.components_
            f['exp_dirichlet_component_'] = self.model.exp_dirichlet_component_
            f['n_batch_iter_'] = np.int_(self.model.n_batch_iter_)
            f['n_features_in_'] = self.model.n_features_in_
            f['n_iter_'] = np.int_(self.model.n_iter_)
            f['bound_'] = np.float_(self.model.bound_)
            f['doc_topic_prior_'] = np.float_(self.model.doc_topic_prior_)
            f['topic_word_prior_'] = np.float_(self.model.topic_word_prior_)

            f.close()

    def get_gene_weights(self):
        """
        get feature(gene) weights

        :return: dataframe contains feature(gene) weights; genes are indexes and topics are columns
        :rtype: pandas dataframe
        """
        gene_weights = pd.DataFrame(np.transpose(self.model.components_),
                                    columns=[f'{self.name}_Topic_{i + 1}' for i in
                                             range(self.model.components_.shape[0])],
                                    index=self.get_feature_name())

        return gene_weights

    def get_ranked_gene_weight(self):
        """
        get sorted feature(gene) weights. each value is gene and weights on each topics

        :return: dataframe contains feature(gene) and their weights; ranks are indexes and topics are columns
        :rtype: pandas dataframe
        """
        gene_weights = pd.DataFrame(np.transpose(self.model.components_),
                                    columns=[f'{self.name}_Topic_{i + 1}' for i in
                                             range(self.model.components_.shape[0])],
                                    index=self.get_feature_name())
        ranked_gene_weights = pd.DataFrame(columns=[f'{self.name}_Topic_{i + 1}' for i in
                                                    range(self.model.components_.shape[0])],
                                           index=range(len(self.get_feature_name())))
        for col in gene_weights.columns:
            tmp = gene_weights.sort_values(by=col, ascending=False)
            tmp[col] = ";" + tmp[col].astype(str)
            tmp[col] = tmp.index.tolist() + tmp[col].astype(str)
            ranked_gene_weights[col] = tmp[col].values
        return ranked_gene_weights

    def get_top_model_attributes(self):
        """
        get top model attributes to be able to make sklearn.decomposition.LatentDirichletAllocation


        :return: three data frame which the first one is components, the second one is exp_dirichlet_component and
        the last one is combining the rest of LDA attributes which put them to gather as a dataframe
        :rtype: pandas dataframe, pandas dataframe, pandas dataframe
        """
        feature = self.get_feature_name()

        components = pd.DataFrame(self.model.components_,
                                  index=[f"Topic_{i + 1}" for i in range(self.N)],
                                  columns=feature)

        exp_dirichlet_component = pd.DataFrame(self.model.exp_dirichlet_component_,
                                               index=[f"Topic_{i + 1}" for i in range(self.N)],
                                               columns=feature)

        others = pd.DataFrame(
            index=[f"Topic_{i + 1}" for i in range(self.N)],
            columns=["n_batch_iter",
                     "n_features_in",
                     "n_iter",
                     "bound",
                     "doc_topic_prior",
                     "topic_word_prior"])

        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "n_batch_iter"] = self.model.n_batch_iter_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "n_features_in"] = self.model.n_features_in_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "n_iter"] = self.model.n_iter_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "bound"] = self.model.bound_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "doc_topic_prior"] = self.model.doc_topic_prior_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "topic_word_prior"] = self.model.topic_word_prior_

        return components, exp_dirichlet_component, others

    def gene_weight_rank_heatmap(self,
                                 genes=None,
                                 topics=None,
                                 show_rank=True,
                                 scale=None,
                                 save=True,
                                 show=True,
                                 figsize=None,
                                 file_format="pdf",
                                 file_name="gene_weight_rank_heatmap"):
        """
        plot selected genes weights and their ranks in selected topics

        :param genes: list of genes you want to see their weights (default: all genes)
        :type genes: list
        :param topics: list of topics
        :type topics: list
        :param show_rank: indicate if you want to show the rank of significant genes or not (default: True)
        :type show_rank: bool
        :param scale: indicate if you want to plot as log2, log10 or not (default: None which show actual value) other options is log2 and log10
        :scale scale: str
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param figsize: indicate the size of plot (default: (10 * (len(category) + 1), 10))
        :type figsize: tuple of int
        :param file_format: indicate the format of plot (default: pdf)
        :type file_format: str
        :param file_name: name and path of the plot use for save (default: gene_weight_rank_heatmap)
        :type file_name: str
        """
        if genes is None:
            genes = self.get_feature_name()
        elif not (set(genes) & set(self.get_feature_name())) == set(genes):
            sys.exit("some/all of genes are not part of topModel!")

        if topics is None:
            topics = [f'{self.name}_Topic_{i + 1}' for i in range(self.model.components_.shape[0])]
        elif not (set(topics) & set(
                [f'{self.name}_Topic_{i + 1}' for i in range(self.model.components_.shape[0])])) == set(topics):
            sys.exit("some/all of topics are not part of topModel!")

        if scale not in ["log2", "log10", None]:
            sys.exit("scale is not valid!")

        if figsize is None:
            figsize = (self.N * 1.5, len(genes))

        gene_weights = self.get_gene_weights()
        gene_weights = gene_weights.loc[:, topics]
        gene_rank = gene_weights.loc[genes, topics]
        for topic in topics:
            tmp = gene_weights.sort_values([topic], ascending=False)
            tmp['rank'] = range(1, tmp.shape[0] + 1)
            gene_rank.loc[genes, topic] = tmp.loc[genes, 'rank'].values

        gene_weights = self.get_gene_weights()
        gene_weights = gene_weights.loc[genes, topics]

        gene_weights = gene_weights.reindex(genes)
        gene_weights = gene_weights.reindex(topics, axis="columns")

        gene_rank = gene_rank.reindex(genes)
        gene_rank = gene_rank.reindex(topics, axis="columns")
        gene_rank = gene_rank.astype(int)
        gene_rank[gene_weights <= self.N] = 0
        gene_rank = gene_rank.astype(str)
        gene_rank[gene_rank == "0"] = ""

        fig, ax = plt.subplots(figsize=figsize,
                               facecolor='white')

        if scale == "log2":
            gene_weights = gene_weights.applymap(np.log2)
        if scale == "log10":
            gene_weights = gene_weights.applymap(np.log10)
        if scale is None:
            scale = "no"

        if show_rank:
            sns.heatmap(gene_weights,
                        cmap='viridis',
                        annot=gene_rank,
                        linewidths=.5,
                        fmt='',
                        ax=ax)
            ax.set_title(f'gene weights with rank ({scale} scale)')
        else:
            sns.heatmap(gene_weights,
                        cmap='viridis',
                        linewidths=.5,
                        ax=ax)
            ax.set_title('gene weights')

        if save:
            plt.savefig(f"{file_name}.{file_format}")
        if show:
            plt.show()
        else:
            plt.close()

    def MA_plot(self,
                topic1,
                topic2,
                size=None,
                pseudocount=1,
                threshold=1,
                cutoff=2,
                consistency_correction=1.4826,
                #topN=None,
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
        :param size: table contains size of dot for each genes (genes are index)
        :type size: pandas dataframe
        :param pseudocount: pseudocount that you want to add (default: 1)
        :type pseudocount: float
        :param threshold: threshold to filter genes based on A values (default: 1)
        :type threshold: float
        :param cutoff: cutoff for categorized genes by modified z-score (default: 2)
        :type cutoff: float
        :param consistency_correction: the factor converts the MAD to the standard deviation for a given distribution. The default value (1.4826) is the conversion factor if the underlying data is normally distributed
        :type consistency_correction: float
        :param topN: number of genes to be consider for calculating z-score based on the A value (if it's none is gonna be avarage of # genes in both topics with weights above threshold
        :type topN: int
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
        gene_weights = self.get_gene_weights().copy(deep=True)
        topic1 = gene_weights[topic1]
        topic2 = gene_weights[topic2]

        gene_zscore = MA_plot(topic1,
                              topic2,
                              size=size,
                              pseudocount=pseudocount,
                              threshold=threshold,
                              cutoff=cutoff,
                              consistency_correction=consistency_correction,
                              #topN=topN,
                              labels=labels,
                              save=save,
                              show=show,
                              file_format=file_format,
                              file_name=file_name)

        return gene_zscore

    def save_topModel(self, name=None, save_path="", file_format='pickle'):
        """
            save TopModel class as a pickle/HDF5 file

            :param name: name of the file (default: topModel_TopModel.name)
            :type name: str
            :param save_path: directory you want to use to save pickle file (default is saving near script)
            :type save_path: str
            :param file_format: format of the file you want to save (option: pickle (default), HDF5)
            :type file_format: str
        """
        if file_format not in ['pickle', 'HDF5']:
            sys.exit(f"{file_format} is not correct! It should be 'pickle' or 'HDF5'.")
        if name is None:
            name = f"topModel_{self.name}"

        if file_format == "pickle":
            print(f"Saving topModel as {name}.p")

            picklefile = open(f"{save_path}{name}.p", "wb")
            pickle.dump(self, picklefile)
            picklefile.close()

        if file_format == "HDF5":
            print(f"Saving topModel as {name}.h5")

            f = h5py.File(f"{name}.h5", "w")
            # model
            model = f.create_group("model")
            model['components_'] = self.model.components_
            model['exp_dirichlet_component_'] = self.model.exp_dirichlet_component_
            model['n_batch_iter_'] = np.int_(self.model.n_batch_iter_)
            model['n_features_in_'] = self.model.n_features_in_
            model['n_iter_'] = np.int_(self.model.n_iter_)
            model['bound_'] = np.float_(self.model.bound_)
            model['doc_topic_prior_'] = np.float_(self.model.doc_topic_prior_)
            model['topic_word_prior_'] = np.float_(self.model.topic_word_prior_)

            # topics
            topics = f.create_group("topics")
            for topic in self.topics.keys():
                topic_gp = topics.create_group(self.topics[topic].id)
                topic_gp['id'] = np.string_(self.topics[topic].id)
                topic_gp['name'] = np.string_(self.topics[topic].name)
                topic_gp['gene_weights'] = self.topics[topic].gene_weights
                gene_information = self.topics[topic].gene_information.copy(deep=True)
                gene_information.reset_index(inplace=True)
                gene_information = gene_information.T.reset_index().T
                topic_gp['gene_information'] = np.array(gene_information)
                topic_information = self.topics[topic].topic_information.copy(deep=True)
                topic_information.reset_index(inplace=True)
                topic_information = topic_information.T.reset_index().T
                topic_gp['topic_information'] = np.array(topic_information)

            f['name'] = np.string_(self.name)
            f['N'] = np.int_(self.N)

            f.close()

