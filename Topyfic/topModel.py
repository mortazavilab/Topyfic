import warnings
import joblib
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings('ignore')

from Topyfic.topic import *


class TopModel:

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
        topic1 = next(iter(self.topics.items()))[1]
        return topic1.gene_information.index.tolist()

    def save_rLDA_model(self, name='rLDA', save_path=""):
        print(f"Saving rLDA model as {name}_{self.N}topics.joblib")

        joblib.dump(self.rLDA, f"{save_path}{name}_{self.N}topics.joblib", compress=3)

    def get_gene_weights(self):
        gene_weights = pd.DataFrame(np.transpose(self.rLDA.components_),
                                    columns=[f'{self.name}_Topic{i + 1}' for i in
                                             range(self.rLDA.components_.shape[0])],
                                    index=self.get_feature_name())

        return gene_weights

    def get_top_models_attributes(self):
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
        print(f"Saving topModel as {name}.p")

        picklefile = open(f"{save_path}{name}.p", "wb")
        pickle.dump(self, picklefile)
        picklefile.close()
