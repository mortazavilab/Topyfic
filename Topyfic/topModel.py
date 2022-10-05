import warnings
import joblib
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

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
    :param rLDA: store reproducible LDA model
    :type rLDA: sklearn.decomposition.LatentDirichletAllocation
    """
    def __init__(self,
                 name,
                 N,
                 gene_weights,
                 rlda=None):
        self.topics = dict()
        for i in range(N):
            topic_gene_weights = pd.DataFrame(gene_weights.iloc[:, i].values,
                                              index=gene_weights.index,
                                              columns=[f"Topic_{i + 1}"])
            self.topics[f"Topic_{i + 1}"] = Topic(topic_id=f"Topic_{i + 1}",
                                                  topic_gene_weights=topic_gene_weights)
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
                                    columns=[f'{self.name}_Topic{i + 1}' for i in
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
                                    columns=[f'{self.name}_Topic{i + 1}' for i in
                                             range(self.rLDA.components_.shape[0])],
                                    index=self.get_feature_name())
        ranked_gene_weights = pd.DataFrame(columns=[f'{self.name}_Topic{i + 1}' for i in
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
        :return: three data frame which the first one is components_, vthe second one is exp_dirichlet_component_ and
        the last one is combining the rest of LDA attributes which put them to gather as a dataframe
        :rtype: pandas dataframe, pandas dataframe, pandas dataframe
        """
        feature = self.get_feature_name()

        components = pd.DataFrame(self.rLDA.components_,
                                  index=[f"Topic{i + 1}" for i in range(self.N)],
                                  columns=feature)

        exp_dirichlet_component = pd.DataFrame(self.rLDA.exp_dirichlet_component_,
                                               index=[f"Topic{i + 1}" for i in range(self.N)],
                                               columns=feature)

        others = pd.DataFrame(
            index=[f"Topic{i + 1}" for i in range(self.N)],
            columns=["n_batch_iter",
                     "n_features_in",
                     "n_iter",
                     "bound",
                     "doc_topic_prior",
                     "topic_word_prior"])

        others.loc[[f"Topic{i + 1}" for i in range(self.N)], "n_batch_iter"] = self.rLDA.n_batch_iter_
        others.loc[[f"Topic{i + 1}" for i in range(self.N)], "n_features_in"] = self.rLDA.n_features_in_
        others.loc[[f"Topic{i + 1}" for i in range(self.N)], "n_iter"] = self.rLDA.n_iter_
        others.loc[[f"Topic{i + 1}" for i in range(self.N)], "bound"] = self.rLDA.bound_
        others.loc[[f"Topic{i + 1}" for i in range(self.N)], "doc_topic_prior"] = self.rLDA.doc_topic_prior_
        others.loc[[f"Topic{i + 1}" for i in range(self.N)], "topic_word_prior"] = self.rLDA.topic_word_prior_

        return components, exp_dirichlet_component, others

    def save_topModel(self, name="topModel", save_path=""):
        """
        save TopModel class as a pickle file
        :param name: name of the pickle file (default: topModel)
        :type name: str
        :param save_path: directory you want to use to save pickle file (default is saving near script)
        :type save_path: str
        """
        print(f"Saving topModel as {name}.p")

        picklefile = open(f"{save_path}{name}.p", "wb")
        pickle.dump(self, picklefile)
        picklefile.close()
