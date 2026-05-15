from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from Topyfic.lda_state import LDAState


@dataclass(frozen=True)
class LDAFitResult:
    model: Any
    document_topic_matrix: np.ndarray


class LDABackend(ABC):
    name = "unknown"

    def __init__(self, **options):
        self.options = dict(options)
        self._prepared_matrix_cache = {}

    def clear_runtime_caches(self):
        self._prepared_matrix_cache.clear()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def transform(self, model, data_matrix):
        raise NotImplementedError()

    @abstractmethod
    def get_state(self, model, feature_names, topic_names=None):
        raise NotImplementedError()

    @abstractmethod
    def model_from_state(self, state: LDAState):
        raise NotImplementedError()