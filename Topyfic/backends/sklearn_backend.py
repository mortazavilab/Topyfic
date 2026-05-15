from __future__ import annotations

from sklearn.decomposition import LatentDirichletAllocation

from Topyfic.backends.base import LDABackend, LDAFitResult
from Topyfic.lda_state import LDAState


class SklearnLDABackend(LDABackend):
    name = "sklearn"

    def fit(self,
            *,
            data_matrix,
            n_components,
            random_state,
            learning_method="online",
            batch_size=1000,
            max_iter=10,
            n_jobs=None,
            **kwargs):
        model = LatentDirichletAllocation(
            n_components=n_components,
            random_state=random_state,
            learning_method=learning_method,
            batch_size=batch_size,
            max_iter=max_iter,
            n_jobs=n_jobs,
            **kwargs,
        )
        document_topic_matrix = model.fit_transform(data_matrix)

        return LDAFitResult(model=model, document_topic_matrix=document_topic_matrix)

    def transform(self, model, data_matrix):
        return model.transform(data_matrix)

    def get_state(self, model, feature_names, topic_names=None):
        return LDAState.from_sklearn_model(
            model,
            feature_names=feature_names,
            topic_names=topic_names,
        )

    def model_from_state(self, state: LDAState):
        return state.to_sklearn_model()