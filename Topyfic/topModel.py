import sys
import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_context('paper')
warnings.filterwarnings('ignore')

from Topyfic.topic import *


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
    :param rlda: store reproducible LDA model
    :type rlda: sklearn.decomposition.LatentDirichletAllocation

    """

    def __init__(self,
                 name,
                 N,
                 gene_weights,
                 gene_information=None,
                 rlda=None):
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
        self.rLDA = rlda

    def get_feature_name(self):
        """
        get feature(gene) name

        :return: list of feature(gene) name
        :rtype: list
        """
        topic1 = next(iter(self.topics.items()))[1]
        return topic1.gene_information.index.tolist()

    def save_rLDA_model(self, name='rLDA', save_path=""):
        """
        save Train class as a pickle file

        :param name: name of the pickle file (default: rLDA)
        :type name: str
        :param save_path: directory you want to use to save pickle file (default is saving near script)
        :type save_path: str
        """
        print(f"Saving rLDA model as {name}_{self.N}topics.joblib")

        joblib.dump(self.rLDA, f"{save_path}{name}_{self.N}topics.joblib", compress=3)

    def get_gene_weights(self):
        """
        get feature(gene) weights

        :return: dataframe contains feature(gene) weights; genes are indexes and topics are columns
        :rtype: pandas dataframe
        """
        gene_weights = pd.DataFrame(np.transpose(self.rLDA.components_),
                                    columns=[f'{self.name}_Topic_{i + 1}' for i in
                                             range(self.rLDA.components_.shape[0])],
                                    index=self.get_feature_name())

        return gene_weights

    def get_ranked_gene_weight(self):
        """
        get sorted feature(gene) weights. each value is gene and weights on each topics

        :return: dataframe contains feature(gene) and their weights; ranks are indexes and topics are columns
        :rtype: pandas dataframe
        """
        gene_weights = pd.DataFrame(np.transpose(self.rLDA.components_),
                                    columns=[f'{self.name}_Topic_{i + 1}' for i in
                                             range(self.rLDA.components_.shape[0])],
                                    index=self.get_feature_name())
        ranked_gene_weights = pd.DataFrame(columns=[f'{self.name}_Topic_{i + 1}' for i in
                                                    range(self.rLDA.components_.shape[0])],
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

        components = pd.DataFrame(self.rLDA.components_,
                                  index=[f"Topic_{i + 1}" for i in range(self.N)],
                                  columns=feature)

        exp_dirichlet_component = pd.DataFrame(self.rLDA.exp_dirichlet_component_,
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

        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "n_batch_iter"] = self.rLDA.n_batch_iter_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "n_features_in"] = self.rLDA.n_features_in_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "n_iter"] = self.rLDA.n_iter_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "bound"] = self.rLDA.bound_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "doc_topic_prior"] = self.rLDA.doc_topic_prior_
        others.loc[[f"Topic_{i + 1}" for i in range(self.N)], "topic_word_prior"] = self.rLDA.topic_word_prior_

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
            topics = [f'{self.name}_Topic_{i + 1}' for i in range(self.rLDA.components_.shape[0])]
        elif not (set(topics) & set(
                [f'{self.name}_Topic_{i + 1}' for i in range(self.rLDA.components_.shape[0])])) == set(topics):
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
        gene_weights = self.get_gene_weights()
        topic1 = gene_weights[topic1]
        topic2 = gene_weights[topic2]

        topic1 += pseudocount
        topic2 += pseudocount

        A = (np.log2(topic1) + np.log2(topic2)) / 2
        M = np.log2(topic1) - np.log2(topic2)

        gene_zscore = pd.concat([A, M], axis=1)
        gene_zscore.columns = ["A", "M"]
        gene_zscore = gene_zscore[gene_zscore.A > threshold]

        gene_zscore['mod_zscore'], mad = TopModel.modified_zscore(gene_zscore['M'], consistency_correction=consistency_correction)

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

    @staticmethod
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

    def save_topModel(self, name=None, save_path=""):
        """
        save TopModel class as a pickle file

        :param name: name of the pickle file (default: topModel_TopModel.name)
        :type name: str
        :param save_path: directory you want to use to save pickle file (default is saving near script)
        :type save_path: str
        """
        if name is None:
            name = f"topModel_{self.name}"
        print(f"Saving topModel as {name}.p")

        picklefile = open(f"{save_path}{name}.p", "wb")
        pickle.dump(self, picklefile)
        picklefile.close()
