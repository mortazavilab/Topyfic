from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


OTHER_COLUMNS = [
    'n_batch_iter',
    'n_features_in',
    'n_iter',
    'bound',
    'doc_topic_prior',
    'topic_word_prior',
]


@dataclass(frozen=True)
class LDAState:
    components: pd.DataFrame
    exp_dirichlet_component: pd.DataFrame
    n_batch_iter: int
    n_features_in: int
    n_iter: int
    bound: float
    doc_topic_prior: float
    topic_word_prior: float

    @classmethod
    def from_sklearn_model(cls, model, feature_names, topic_names=None):
        if topic_names is None:
            topic_names = [f'Topic_{index + 1}' for index in range(model.components_.shape[0])]

        components = pd.DataFrame(model.components_, index=topic_names, columns=feature_names)
        exp_dirichlet_component = pd.DataFrame(
            model.exp_dirichlet_component_,
            index=topic_names,
            columns=feature_names,
        )

        return cls(
            components=components,
            exp_dirichlet_component=exp_dirichlet_component,
            n_batch_iter=int(model.n_batch_iter_),
            n_features_in=int(model.n_features_in_),
            n_iter=int(model.n_iter_),
            bound=float(model.bound_),
            doc_topic_prior=float(model.doc_topic_prior_),
            topic_word_prior=float(model.topic_word_prior_),
        )

    @classmethod
    def from_frames(cls, components, exp_dirichlet_component, others):
        if isinstance(others, pd.Series):
            others_row = others
        else:
            others_row = others.iloc[0]

        return cls(
            components=components.copy(deep=True),
            exp_dirichlet_component=exp_dirichlet_component.copy(deep=True),
            n_batch_iter=int(others_row['n_batch_iter']),
            n_features_in=int(others_row['n_features_in']),
            n_iter=int(others_row['n_iter']),
            bound=float(others_row['bound']),
            doc_topic_prior=float(others_row['doc_topic_prior']),
            topic_word_prior=float(others_row['topic_word_prior']),
        )

    def to_frames(self):
        others = pd.DataFrame(index=self.components.index, columns=OTHER_COLUMNS)
        others.loc[:, 'n_batch_iter'] = self.n_batch_iter
        others.loc[:, 'n_features_in'] = self.n_features_in
        others.loc[:, 'n_iter'] = self.n_iter
        others.loc[:, 'bound'] = self.bound
        others.loc[:, 'doc_topic_prior'] = self.doc_topic_prior
        others.loc[:, 'topic_word_prior'] = self.topic_word_prior

        return (
            self.components.copy(deep=True),
            self.exp_dirichlet_component.copy(deep=True),
            others,
        )

    def to_sklearn_model(self):
        model = LatentDirichletAllocation(n_components=self.components.shape[0])
        model.components_ = self.components.values
        model.exp_dirichlet_component_ = self.exp_dirichlet_component.values
        model.n_batch_iter_ = self.n_batch_iter
        model.n_features_in_ = self.n_features_in
        model.n_iter_ = self.n_iter
        model.bound_ = self.bound
        model.doc_topic_prior_ = self.doc_topic_prior
        model.topic_word_prior_ = self.topic_word_prior

        return model