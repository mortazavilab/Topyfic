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
import h5py

from Topyfic.topModel import TopModel

warnings.filterwarnings("ignore")


class Train:
    """
    A class used to train reproducible latent dirichlet allocation (rLDA) model

    :param name: name of the Train class
    :type name: str
    :param k: number of topics to learn one LDA model using sklearn package
    :type k: int
    :param n_runs: number of run to define rLDA model (default: 100)
    :type n_runs: int
    :param random_state_range: list of random state, we used to run LDA models (default: range(n_runs))
    :type random_state_range: list of int
    :param top_models: list of TopModel class to save all LDA models
    :type top_models: list of TopModel
    """

    def __init__(self,
                 name,
                 k,
                 n_runs=100,
                 random_state_range=None):

        if random_state_range is None:
            random_state_range = range(n_runs)
        elif n_runs != len(random_state_range):
            sys.exit("number of runs and length of random state did not match!")

        self.name = name
        self.k = k
        self.n_runs = n_runs
        self.random_state_range = random_state_range
        self.top_models = []

    def combine_LDA_models(self, data, single_trains=[]):
        """
        combine single top_model

        :param data: data you used to learn model
        :type data: anndata
        :param single_trains: list of single train object
        :type single_trains: list
        """
        for i in range(len(single_trains)):
            gene_weights = pd.DataFrame(np.transpose(single_trains[i].top_models[0].model.components_),
                                        columns=[f'Topic{j + 1}_R{self.random_state_range[i]}' for j in range(self.k)],
                                        index=data.var.index.tolist())
            TopModel_lda_model = TopModel(name=f"{self.name}_{self.random_state_range[i]}",
                                          N=gene_weights.shape[1],
                                          gene_weights=gene_weights,
                                          model=single_trains[i].top_models[0].model)
            self.top_models.append(TopModel_lda_model)

    def make_single_LDA_model(self, data, random_state, name, learning_method, batch_size, max_iter, n_jobs, kwargs):
        """
        train simple LDA model using sklearn package and embed it to TopModel class


        :param name: name of LDA model
        :type name: str
        :param data: processed expression data along with cells and genes/region information
        :type data: anndata
        :param random_state: Pass an int for reproducible results across multiple function calls
        :type random_state: int
        :param max_iter: The maximum number of passes over the training data (aka epochs) (default = 10)
        :type max_iter: int
        :param batch_size: Number of documents to use in each EM iteration. Only used in online learning. (default = 1000)
        :type batch_size: int
        :param learning_method: Method used to update _component. {‘batch’, ‘online’} (default=’online’)
        :type learning_method: str
        :param n_jobs: The number of jobs to use in the E-step. None means 1 unless in a `joblib.parallel_backend <https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend>`_ context. -1 means using all processors.  See `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_ for more details. (default = None)
        :type n_jobs: int

        :return: LDA model embedded in TopModel class
        :rtype: TopModel
        """
        lda_model = LatentDirichletAllocation(n_components=self.k,
                                              random_state=random_state,
                                              learning_method=learning_method,
                                              batch_size=batch_size,
                                              max_iter=max_iter,
                                              n_jobs=n_jobs,
                                              **kwargs)

        lda_model.fit_transform(data.to_df().to_numpy())

        gene_weights = pd.DataFrame(np.transpose(lda_model.components_),
                                    columns=[f'Topic{i + 1}_R{random_state}' for i in range(self.k)],
                                    index=data.var.index.tolist())

        TopModel_lda_model = TopModel(name=f"{name}_{random_state}",
                                      N=gene_weights.shape[1],
                                      gene_weights=gene_weights,
                                      model=lda_model)

        return TopModel_lda_model

    def run_LDA_models(self, data, learning_method="online", batch_size=1000, max_iter=10, n_jobs=None, n_thread=1, **kwargs):
        """
        train LDA models


        :param max_iter: The maximum number of passes over the training data (aka epochs) (default = 10)
        :type max_iter: int
        :param batch_size: Number of documents to use in each EM iteration. Only used in online learning. (default = 1000)
        :type batch_size: int
        :param learning_method: Method used to update _component. {‘batch’, ‘online’} (default=’online’)
        :type learning_method: str
        :param data: expression data embedded in anndata format use to train LDA model
        :type data: anndata
        :param n_jobs: The number of jobs to use in the E-step. None means 1 unless in a `joblib.parallel_backend <https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend>`_ context. -1 means using all processors.  See `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_ for more details. (default = None)
        :type n_jobs: int
        :param n_thread: number of threads you used to learn LDA models (default=1)
        :type n_thread: int
        :param **kwargs: other parameter in sklearn.decomposition.LatentDirichletAllocation function (more info: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
        :type **kwargs: dict

        :return: None
        :rtype: None
        """
        if n_thread == 1:
            self.top_models = []
            for random_state in self.random_state_range:
                self.top_models.append(self.make_single_LDA_model(data, random_state, self.name, learning_method, batch_size, max_iter, n_jobs, kwargs))

        self.top_models = Pool(processes=n_thread).starmap(self.make_single_LDA_model,
                                                           zip(repeat(data), self.random_state_range, repeat(self.name),
                                                               repeat(learning_method), repeat(batch_size),
                                                               repeat(max_iter), repeat(n_jobs), repeat(kwargs)))
        print(f"{self.n_runs} LDA models with {self.k} topics learned\n")

    def make_LDA_models_attributes(self):
        """
        make LDA attributes by combining all single LDA model attributes which you need to define LDA model (sklearn.decomposition.LatentDirichletAllocation)


        :return: three data frame which the first one is gathering all components from all LDA runs,
        the second one is exp_dirichlet_component from all LDA runs and
        the last one is combining the rest of LDA attributes which put them to gather as a dataframe
        :rtype: pandas dataframe, pandas dataframe, pandas dataframe

        """
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

        count = 0
        for random_state in self.random_state_range:
            components, exp_dirichlet_component, others = self.top_models[count].get_top_model_attributes()

            all_components.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], :] = components.values

            all_exp_dirichlet_component.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], :] = exp_dirichlet_component.values

            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "n_batch_iter"] = others.n_batch_iter.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "n_features_in"] = others.n_features_in.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "n_iter"] = others.n_iter.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "bound"] = others.bound.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "doc_topic_prior"] = others.doc_topic_prior.values
            all_others.loc[[f"Topic{i + 1}_R{random_state}" for i in range(self.k)], "topic_word_prior"] = others.topic_word_prior.values

            count = count + 1

        return all_components, all_exp_dirichlet_component, all_others

    def save_train(self, name=None, save_path="", file_format='pickle'):
        """
            save Train class as a pickle file

            :param name: name of the pickle file (default is train_Train.name)
            :type name: str
            :param save_path: directory you want to use to save pickle file (default is saving near script)
            :type save_path: str
            :param file_format: format of the file you want to save (option: pickle (default), HDF5)
            :type file_format: str
        """
        if file_format not in ['pickle', 'HDF5']:
            sys.exit(f"{file_format} is not correct! It should be 'pickle' or 'HDF5'.")
        if name is None:
            name = f"train_{self.name}"

        if file_format == "pickle":
            print(f"Saving train as {name}.p")

            picklefile = open(f"{save_path}{name}.p", "wb")
            pickle.dump(self, picklefile)
            picklefile.close()

        if file_format == "HDF5":
            print(f"Saving train as {name}.h5")

            f = h5py.File(f"{name}.h5", "w")

            # models
            models = f.create_group("models")
            for i in range(len(self.top_models)):
                model = models.create_group(str(i))
                model['components_'] = self.top_models[i].model.components_
                model['exp_dirichlet_component_'] = self.top_models[i].model.exp_dirichlet_component_
                model['n_batch_iter_'] = np.int_(self.top_models[i].model.n_batch_iter_)
                model['n_features_in_'] = self.top_models[i].model.n_features_in_
                model['n_iter_'] = np.int_(self.top_models[i].model.n_iter_)
                model['bound_'] = np.float_(self.top_models[i].model.bound_)
                model['doc_topic_prior_'] = np.float_(self.top_models[i].model.doc_topic_prior_)
                model['topic_word_prior_'] = np.float_(self.top_models[i].model.topic_word_prior_)

            f['name'] = np.string_(self.name)
            f['k'] = np.int_(self.k)
            f['n_runs'] = np.int_(self.n_runs)
            f['random_state_range'] = np.array(list(self.random_state_range))

            f.close()
