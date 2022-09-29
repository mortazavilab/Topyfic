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

from Topyfic.topic import *
from Topyfic.topModel import *

warnings.filterwarnings("ignore")


class Train:
    """
    A class used to train reproducible latent dirichlet allocation (rLDA) model

    :param name: name of the Train class
    :type name: str
    :param k: number of topics to learn one LDA model using sklearn package (default: 50)
    :type k: int
    :param n_runs: number of run to define rLDA model (default: 100)
    :type n_runs: int
    :param random_state_range: list of random state, we used to run LDA models (default: range(n_runs))
    :type random_state_range: list of int
    :param top_models: list of top_models which each of them save the LDA model with specific random_state
    :type top_models: list of TopModel
    """

    def __init__(self,
                 name,
                 k=50,
                 n_runs=100,
                 random_state_range=None):

        if random_state_range is None:
            random_state_range = range(n_runs)
        else:
            n_runs = len(random_state_range)

        self.name = name
        self.k = k
        self.n_runs = n_runs
        self.random_state_range = random_state_range
        self.top_models = []

    def make_single_LDA_model(self, data, random_state):
        """
        train simple LDA model using sklearn package and embed it to TopModel class

        :param data: processed gene count data along with cells and genes information
        :type data: anndata
        :param random_state: Pass an int for reproducible results across multiple function calls
        :type random_state: int

        :return: LDA model embeded in TopModel class
        :rtype: TopModel
        """
        lda_model = LatentDirichletAllocation(n_components=self.k,
                                              learning_method="online",
                                              random_state=random_state)

        lda_model.fit_transform(data.to_df().to_numpy())

        gene_weights = pd.DataFrame(np.transpose(lda_model.components_),
                                    columns=[f'Topic{i + 1}_R{random_state}' for i in range(self.k)],
                                    index=data.var.index.tolist())

        TopModel_lda_model = TopModel(N=gene_weights.shape[1], gene_weights=gene_weights, rlda=lda_model)

        return TopModel_lda_model

    def run_LDA_models(self, data, n_thread=None):
        """
        train LDA model
        :param data:
        :param n_thread:
        :return:
        """
        if n_thread is None:
            n_thread = self.n_runs

        self.top_models = Pool(processes=n_thread).starmap(self.make_single_LDA_model,
                                                           zip(repeat(data), self.random_state_range))
        print(f"{self.n_runs} LDA models with {self.k} topics learned\n")

    def make_LDA_models_attributes(self):
        feature = self.top_models[0].get_feature_name()

        all_components = pd.DataFrame(
            index=[f"Topic{i + 1}_R{j}" for j in self.random_state_range for i in range(self.k)],
            columns=feature)

        all_exp_dirichlet_component = pd.DataFrame(
            index=[f"Topic{i + 1}_R{j}" for j in self.random_state_range for i in range(self.k)],
            columns=feature)

        all_others = pd.DataFrame(
            index=[f"Topic{i + 1}_R{j}" for j in self.random_state_range for i in range(self.k)],
            columns=["n_batch_iter",
                     "n_features_in",
                     "n_iter",
                     "bound",
                     "doc_topic_prior",
                     "topic_word_prior"])

        for random_state in self.random_state_range:
            components, exp_dirichlet_component, others = self.top_models[random_state].get_top_models_attributes()

            all_components.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], :] = components.values

            all_exp_dirichlet_component.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], :] = exp_dirichlet_component.values

            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "n_batch_iter"] = others.n_batch_iter.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "n_features_in"] = others.n_features_in.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "n_iter"] = others.n_iter.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "bound"] = others.bound.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "doc_topic_prior"] = others.doc_topic_prior.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "topic_word_prior"] = others.topic_word_prior.values

        return all_components, all_exp_dirichlet_component, all_others

    def save_train(self, name=None, save_path=""):
        if name is None:
            name = f"train_{self.name}"

        print(f"Saving train class as {name}.p")

        picklefile = open(f"{save_path}{name}.p", "wb")
        pickle.dump(self, picklefile)
        picklefile.close()
