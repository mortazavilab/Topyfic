from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Topyfic.backends.base import LDABackend
from Topyfic.lda_state import LDAState

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent
    torch = None


@dataclass
class TorchLDAModel:
    lambda_parameter: "torch.Tensor"
    exp_dirichlet_component_tensor: "torch.Tensor"
    n_batch_iter_: int
    n_features_in_: int
    n_iter_: int
    bound_: float
    doc_topic_prior_: float
    topic_word_prior_: float
    device: str
    dtype: str
    backend_name: str = "torch"

    @property
    def components_(self):
        return self.lambda_parameter.detach().cpu().numpy()

    @property
    def exp_dirichlet_component_(self):
        return self.exp_dirichlet_component_tensor.detach().cpu().numpy()


class TorchLDABackend(LDABackend):
    name = "torch"

    def __init__(self, device="auto", dtype="float32", **options):
        super().__init__(device=device, dtype=dtype, **options)
        self.device = self.resolve_device(device)
        self.dtype = dtype

    @staticmethod
    def is_available():
        return torch is not None

    @staticmethod
    def resolve_device(requested="auto"):
        if requested != "auto":
            return requested

        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _require_torch(self):
        if torch is None:
            raise ImportError("TorchLDABackend requires the optional 'torch' dependency")

    def _torch_dtype(self):
        self._require_torch()
        if self.dtype == "float64":
            return torch.float64
        return torch.float32

    def _to_torch_matrix(self, data_matrix):
        self._require_torch()
        if hasattr(data_matrix, "toarray"):
            data_matrix = data_matrix.toarray()
        elif hasattr(data_matrix, "A"):
            data_matrix = data_matrix.A

        return torch.tensor(
            np.asarray(data_matrix),
            device=self.device,
            dtype=self._torch_dtype(),
        )

    @staticmethod
    def _resolve_prior(value, n_components):
        if value is None:
            return 1.0 / n_components
        return float(value)

    @staticmethod
    def _normalize_rows(matrix):
        row_sums = matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return matrix / row_sums

    def _infer_gamma(self,
                     matrix,
                     lambda_parameter,
                     doc_topic_prior,
                     max_doc_update_iter,
                     mean_change_tol,
                     collect_sufficient_stats):
        n_docs = matrix.shape[0]
        n_components = lambda_parameter.shape[0]
        elogbeta = torch.special.digamma(lambda_parameter) - torch.special.digamma(
            lambda_parameter.sum(dim=1, keepdim=True)
        )
        gamma = torch.empty((n_docs, n_components), device=self.device, dtype=self._torch_dtype())
        sufficient_stats = None
        if collect_sufficient_stats:
            sufficient_stats = torch.zeros_like(lambda_parameter)

        for doc_index in range(n_docs):
            counts = matrix[doc_index]
            nonzero = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
            total_count = counts.sum()

            if nonzero.numel() == 0:
                gamma_doc = torch.full(
                    (n_components,),
                    fill_value=float(doc_topic_prior),
                    device=self.device,
                    dtype=self._torch_dtype(),
                )
                gamma[doc_index] = gamma_doc
                continue

            counts_nonzero = counts.index_select(0, nonzero)
            gamma_doc = torch.full(
                (n_components,),
                fill_value=float(doc_topic_prior) + float(total_count.detach().cpu().item()) / n_components,
                device=self.device,
                dtype=self._torch_dtype(),
            )

            for _ in range(max_doc_update_iter):
                elogtheta = torch.special.digamma(gamma_doc) - torch.special.digamma(gamma_doc.sum())
                log_phi = elogtheta[:, None] + elogbeta.index_select(1, nonzero)
                log_phi = log_phi - torch.logsumexp(log_phi, dim=0, keepdim=True)
                phi = torch.exp(log_phi)
                next_gamma = float(doc_topic_prior) + (phi * counts_nonzero.unsqueeze(0)).sum(dim=1)
                if torch.mean(torch.abs(next_gamma - gamma_doc)) < mean_change_tol:
                    gamma_doc = next_gamma
                    break
                gamma_doc = next_gamma

            gamma[doc_index] = gamma_doc

            if collect_sufficient_stats:
                sufficient_stats[:, nonzero] += phi * counts_nonzero.unsqueeze(0)

        return gamma, sufficient_stats

    def _estimate_bound(self, matrix, document_topic_matrix, lambda_parameter):
        beta = self._normalize_rows(lambda_parameter)
        word_probabilities = torch.matmul(document_topic_matrix, beta).clamp_min(1e-12)
        return float((matrix * torch.log(word_probabilities)).sum().detach().cpu().item())

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
        self._require_torch()

        if learning_method not in {"batch", "online"}:
            raise ValueError("learning_method must be 'batch' or 'online'")

        matrix = self._to_torch_matrix(data_matrix)
        n_components = int(n_components)
        n_features = int(matrix.shape[1])
        doc_topic_prior = self._resolve_prior(kwargs.pop("doc_topic_prior", None), n_components)
        topic_word_prior = self._resolve_prior(kwargs.pop("topic_word_prior", None), n_components)
        max_doc_update_iter = int(kwargs.pop("max_doc_update_iter", 50))
        mean_change_tol = float(kwargs.pop("mean_change_tol", 1e-3))

        rng = np.random.default_rng(random_state)
        lambda_parameter = torch.tensor(
            rng.gamma(shape=100.0, scale=0.01, size=(n_components, n_features)),
            device=self.device,
            dtype=self._torch_dtype(),
        )
        lambda_parameter = lambda_parameter + float(topic_word_prior)

        gamma = None
        for _ in range(int(max_iter)):
            gamma, sufficient_stats = self._infer_gamma(
                matrix=matrix,
                lambda_parameter=lambda_parameter,
                doc_topic_prior=doc_topic_prior,
                max_doc_update_iter=max_doc_update_iter,
                mean_change_tol=mean_change_tol,
                collect_sufficient_stats=True,
            )
            lambda_parameter = sufficient_stats + float(topic_word_prior)

        exp_dirichlet_component = torch.exp(
            torch.special.digamma(lambda_parameter) - torch.special.digamma(lambda_parameter.sum(dim=1, keepdim=True))
        )
        document_topic_matrix = self._normalize_rows(gamma)
        bound = self._estimate_bound(matrix, document_topic_matrix, lambda_parameter)

        model = TorchLDAModel(
            lambda_parameter=lambda_parameter,
            exp_dirichlet_component_tensor=exp_dirichlet_component,
            n_batch_iter_=int(max_iter),
            n_features_in_=n_features,
            n_iter_=int(max_iter),
            bound_=bound,
            doc_topic_prior_=doc_topic_prior,
            topic_word_prior_=topic_word_prior,
            device=self.device,
            dtype=self.dtype,
        )

        return self._fit_result_type()(model=model, document_topic_matrix=document_topic_matrix.detach().cpu().numpy())

    def transform(self, model, data_matrix):
        matrix = self._to_torch_matrix(data_matrix)
        gamma, _ = self._infer_gamma(
            matrix=matrix,
            lambda_parameter=model.lambda_parameter,
            doc_topic_prior=model.doc_topic_prior_,
            max_doc_update_iter=int(self.options.get("max_doc_update_iter", 50)),
            mean_change_tol=float(self.options.get("mean_change_tol", 1e-3)),
            collect_sufficient_stats=False,
        )
        return self._normalize_rows(gamma).detach().cpu().numpy()

    def get_state(self, model, feature_names=None, topic_names=None):
        if isinstance(model, TorchLDAModel):
            if topic_names is None:
                topic_names = [f"Topic_{index + 1}" for index in range(model.components_.shape[0])]
            components = pd.DataFrame(model.components_, index=topic_names, columns=feature_names)
            exp_dirichlet_component = pd.DataFrame(
                model.exp_dirichlet_component_,
                index=topic_names,
                columns=feature_names,
            )
            others = pd.DataFrame(
                {
                    "n_batch_iter": [model.n_batch_iter_],
                    "n_features_in": [model.n_features_in_],
                    "n_iter": [model.n_iter_],
                    "bound": [model.bound_],
                    "doc_topic_prior": [model.doc_topic_prior_],
                    "topic_word_prior": [model.topic_word_prior_],
                }
            )
            return LDAState.from_frames(components, exp_dirichlet_component, others)
        raise TypeError("TorchLDABackend can only export state from TorchLDAModel instances")

    def model_from_state(self, state: LDAState):
        self._require_torch()
        dtype = self._torch_dtype()

        components = torch.tensor(state.components.values, device=self.device, dtype=dtype)
        exp_dirichlet_component = torch.tensor(
            state.exp_dirichlet_component.values,
            device=self.device,
            dtype=dtype,
        )

        return TorchLDAModel(
            lambda_parameter=components,
            exp_dirichlet_component_tensor=exp_dirichlet_component,
            n_batch_iter_=state.n_batch_iter,
            n_features_in_=state.n_features_in,
            n_iter_=state.n_iter,
            bound_=state.bound,
            doc_topic_prior_=state.doc_topic_prior,
            topic_word_prior_=state.topic_word_prior,
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def _fit_result_type():
        from Topyfic.backends.base import LDAFitResult

        return LDAFitResult