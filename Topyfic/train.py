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

from Topyfic.backends import create_lda_backend
from Topyfic.persistence import write_backend_metadata, write_lda_state
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
                 random_state_range=None,
                 backend_name="sklearn",
                 backend_kwargs=None):

        if random_state_range is None:
            random_state_range = range(n_runs)
        elif n_runs != len(random_state_range):
            sys.exit("number of runs and length of random state did not match!")

        self.name = name
        self.k = k
        self.n_runs = n_runs
        self.random_state_range = random_state_range
        self.backend_name = backend_name
        self.backend_kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
        self.top_models = []
        self._backend = None
        self._backend_cache_key = None

    @property
    def backend(self):
        cache_key = (self.backend_name, tuple(sorted(self.backend_kwargs.items())))
        if getattr(self, "_backend", None) is None or getattr(self, "_backend_cache_key", None) != cache_key:
            self._backend = create_lda_backend(self.backend_name, **self.backend_kwargs)
            self._backend_cache_key = cache_key
        return self._backend

    def combine_LDA_models(self, data, single_trains=[]):
        """
        combine single top_model

        :param data: data you used to learn model
        :type data: anndata
        :param single_trains: list of single train object
        :type single_trains: list
        """
        if single_trains:
            backend_names = {single_train.top_models[0].backend_name for single_train in single_trains}
            if len(backend_names) > 1:
                raise ValueError("single_trains must share the same backend before they can be combined")
            self.backend_name = single_trains[0].top_models[0].backend_name
            self.backend_kwargs = dict(single_trains[0].top_models[0].backend_kwargs)
            self._backend = None
            self._backend_cache_key = None

        for i in range(len(single_trains)):
            gene_weights = pd.DataFrame(np.transpose(single_trains[i].top_models[0].model.components_),
                                        columns=[f'Topic{j + 1}_R{self.random_state_range[i]}' for j in range(self.k)],
                                        index=data.var.index.tolist())
            TopModel_lda_model = TopModel(name=f"{self.name}_{self.random_state_range[i]}",
                                          N=gene_weights.shape[1],
                                          gene_weights=gene_weights,
                                          model=single_trains[i].top_models[0].model,
                                          backend_name=single_trains[i].top_models[0].backend_name,
                                          backend_kwargs=single_trains[i].top_models[0].backend_kwargs)
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
        fit_result = self.backend.fit(data_matrix=data.X,
                          n_components=self.k,
                          random_state=random_state,
                          learning_method=learning_method,
                          batch_size=batch_size,
                          max_iter=max_iter,
                          n_jobs=n_jobs,
                          **kwargs)

        lda_model = fit_result.model

        gene_weights = pd.DataFrame(np.transpose(lda_model.components_),
                                    columns=[f'Topic{i + 1}_R{random_state}' for i in range(self.k)],
                                    index=data.var.index.tolist())

        TopModel_lda_model = TopModel(name=f"{name}_{random_state}",
                                      N=gene_weights.shape[1],
                                      gene_weights=gene_weights,
                                      model=lda_model,
                                      backend_name=self.backend_name,
                                      backend_kwargs=self.backend_kwargs)

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
        else:
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

            f = h5py.File(os.path.join(save_path, f"{name}.h5"), "w")
            write_backend_metadata(f, self.backend_name, self.backend_kwargs)

            # models
            models = f.create_group("models")
            for i in range(len(self.top_models)):
                random_state = self.random_state_range[i]
                model = models.create_group(str(random_state))
                write_backend_metadata(
                    model,
                    self.top_models[i].backend_name,
                    self.top_models[i].backend_kwargs,
                )
                write_lda_state(model, self.top_models[i].get_backend_state())

            f['name'] = self.name.encode('utf-8')
            f['k'] = int(self.k)
            f['n_runs'] = int(self.n_runs)
            f['random_state_range'] = np.array(list(self.random_state_range))

            f.close()
