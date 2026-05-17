from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse as sp

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


@dataclass(frozen=True)
class TorchSparseBatch:
    rows: "torch.Tensor"
    cols: "torch.Tensor"
    counts: "torch.Tensor"
    row_sums: "torch.Tensor"
    n_docs: int


@dataclass(frozen=True)
class TorchPreparedMatrix:
    batches: list[TorchSparseBatch]
    n_docs: int
    n_features: int


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

    def _document_batch_size(self, n_docs, requested=None):
        if requested is None:
            requested = self.options.get("batch_size", 256)
        return max(1, min(int(requested), int(n_docs)))

    @staticmethod
    def _prepared_matrix_cache_key(data_matrix, batch_size, dtype):
        sparse_nnz = getattr(data_matrix, "nnz", None)
        return (id(data_matrix), tuple(getattr(data_matrix, "shape", ())), sparse_nnz, int(batch_size), dtype)

    def _to_csr_matrix(self, data_matrix):
        if sp.issparse(data_matrix):
            return data_matrix.tocsr()
        return sp.csr_matrix(np.asarray(data_matrix))

    @staticmethod
    def _resolve_prior(value, n_components):
        if value is None:
            return 1.0 / n_components
        return float(value)

    @staticmethod
    def _normalize_rows(matrix):
        row_sums = matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return matrix / row_sums

    @staticmethod
    def _exp_dirichlet_expectation(matrix):
        return torch.exp(
            torch.special.digamma(matrix) - torch.special.digamma(matrix.sum(dim=1, keepdim=True))
        )

    def _prepare_sparse_batches(self, data_matrix, batch_size=None):
        matrix_csr = self._to_csr_matrix(data_matrix)
        n_docs = int(matrix_csr.shape[0])
        batch_size = self._document_batch_size(n_docs=n_docs, requested=batch_size)
        cache_key = self._prepared_matrix_cache_key(data_matrix, batch_size, self.dtype)
        cached = self._prepared_matrix_cache.get(cache_key)
        if cached is not None:
            return cached

        numpy_dtype = np.float32 if self.dtype == "float32" else np.float64
        batches = []

        for batch_start in range(0, n_docs, batch_size):
            batch_end = min(batch_start + batch_size, n_docs)
            batch_csr = matrix_csr[batch_start:batch_end]
            rows = np.repeat(np.arange(batch_csr.shape[0], dtype=np.int64), np.diff(batch_csr.indptr))
            cols = batch_csr.indices.astype(np.int64, copy=False)
            counts = batch_csr.data.astype(numpy_dtype, copy=False)
            row_sums = np.asarray(batch_csr.sum(axis=1)).reshape(-1).astype(numpy_dtype, copy=False)
            batches.append(
                TorchSparseBatch(
                    rows=torch.tensor(rows, device=self.device, dtype=torch.long),
                    cols=torch.tensor(cols, device=self.device, dtype=torch.long),
                    counts=torch.tensor(counts, device=self.device, dtype=self._torch_dtype()),
                    row_sums=torch.tensor(row_sums, device=self.device, dtype=self._torch_dtype()),
                    n_docs=int(batch_csr.shape[0]),
                )
            )

        prepared_matrix = TorchPreparedMatrix(
            batches=batches,
            n_docs=n_docs,
            n_features=int(matrix_csr.shape[1]),
        )
        self._prepared_matrix_cache = {cache_key: prepared_matrix}
        return prepared_matrix

    def _finalize_batch(self,
                        doc_topic_prior,
                        gamma_batch,
                        rows,
                        cols,
                        counts,
                        exp_e_log_beta,
                        collect_sufficient_stats,
                        exp_beta_cols=None,
                        sufficient_stats_buffer=None):
        exp_e_log_theta = self._exp_dirichlet_expectation(gamma_batch)

        if rows.numel() == 0:
            sufficient_stats = None
            if collect_sufficient_stats and sufficient_stats_buffer is None:
                sufficient_stats = torch.zeros_like(exp_e_log_beta)
            return gamma_batch, exp_e_log_theta, sufficient_stats

        exp_theta_rows = exp_e_log_theta.index_select(0, rows)
        if exp_beta_cols is None:
            exp_beta_cols = exp_e_log_beta.index_select(1, cols).transpose(0, 1)
        norm_phi = (exp_theta_rows * exp_beta_cols).sum(dim=1).clamp_min(1e-12)
        weighted_counts = counts / norm_phi

        sufficient_stats = None
        if collect_sufficient_stats:
            topic_word_contrib = weighted_counts.unsqueeze(1) * exp_theta_rows * exp_beta_cols
            sufficient_stats = self._accumulate_sufficient_stats(
                cols=cols,
                topic_word_contrib=topic_word_contrib,
                n_features=exp_e_log_beta.shape[1],
                output_buffer=sufficient_stats_buffer,
            )

        return gamma_batch, exp_e_log_theta, sufficient_stats

    def _accumulate_sufficient_stats(self, cols, topic_word_contrib, n_features, output_buffer=None):
        n_components = topic_word_contrib.shape[1]
        target_buffer = output_buffer
        if output_buffer is None:
            flat_stats = torch.zeros(
                n_components * n_features,
                device=topic_word_contrib.device,
                dtype=topic_word_contrib.dtype,
            )
        else:
            flat_stats = output_buffer.reshape(-1)

        topic_offsets = torch.arange(n_components, device=cols.device, dtype=cols.dtype) * int(n_features)
        flat_indices = cols.unsqueeze(1) + topic_offsets.unsqueeze(0)
        flat_stats.index_add_(0, flat_indices.reshape(-1), topic_word_contrib.reshape(-1))
        if target_buffer is not None:
            return target_buffer
        return flat_stats.reshape(n_components, n_features)

    def _infer_gamma_batch(self,
                           prepared_batch,
                           exp_e_log_beta,
                           doc_topic_prior,
                           max_doc_update_iter,
                           mean_change_tol,
                           collect_sufficient_stats,
                           sufficient_stats_buffer=None):
        n_docs = prepared_batch.n_docs
        n_components = exp_e_log_beta.shape[0]
        if n_docs == 0:
            empty_gamma = torch.empty((0, n_components), device=self.device, dtype=self._torch_dtype())
            empty_stats = None
            if collect_sufficient_stats and sufficient_stats_buffer is None:
                empty_stats = torch.zeros_like(exp_e_log_beta)
            return empty_gamma, empty_stats

        gamma_batch = torch.full(
            (n_docs, n_components),
            fill_value=float(doc_topic_prior),
            device=self.device,
            dtype=self._torch_dtype(),
        )
        gamma_batch = gamma_batch + prepared_batch.row_sums.unsqueeze(1) / float(n_components)

        rows = prepared_batch.rows
        cols = prepared_batch.cols
        counts = prepared_batch.counts
        exp_beta_cols = exp_e_log_beta.index_select(1, cols).transpose(0, 1) if rows.numel() != 0 else None

        if rows.numel() == 0:
            sufficient_stats = None
            if collect_sufficient_stats and sufficient_stats_buffer is None:
                sufficient_stats = torch.zeros_like(exp_e_log_beta)
            return gamma_batch, sufficient_stats

        per_doc_topic = torch.zeros_like(gamma_batch)

        for _ in range(max_doc_update_iter):
            exp_e_log_theta = self._exp_dirichlet_expectation(gamma_batch)
            exp_theta_rows = exp_e_log_theta.index_select(0, rows)
            norm_phi = (exp_theta_rows * exp_beta_cols).sum(dim=1).clamp_min(1e-12)
            weighted_counts = counts / norm_phi
            doc_topic_contrib = weighted_counts.unsqueeze(1) * exp_beta_cols

            per_doc_topic.zero_()
            per_doc_topic.index_add_(0, rows, doc_topic_contrib)

            next_gamma = float(doc_topic_prior) + exp_e_log_theta * per_doc_topic

            if torch.mean(torch.abs(next_gamma - gamma_batch)) < mean_change_tol:
                gamma_batch = next_gamma
                break
            gamma_batch = next_gamma

        gamma_batch, _, sufficient_stats = self._finalize_batch(
            doc_topic_prior=doc_topic_prior,
            gamma_batch=gamma_batch,
            rows=rows,
            cols=cols,
            counts=counts,
            exp_e_log_beta=exp_e_log_beta,
            collect_sufficient_stats=collect_sufficient_stats,
            exp_beta_cols=exp_beta_cols,
            sufficient_stats_buffer=sufficient_stats_buffer,
        )

        return gamma_batch, sufficient_stats

    def _infer_gamma(self,
                     prepared_matrix,
                     lambda_parameter,
                     doc_topic_prior,
                     max_doc_update_iter,
                     mean_change_tol,
                     collect_sufficient_stats,
                     batch_size=None,
                     exp_e_log_beta=None):
        n_docs = prepared_matrix.n_docs
        n_components = lambda_parameter.shape[0]
        gamma = torch.empty((n_docs, n_components), device=self.device, dtype=self._torch_dtype())
        sufficient_stats = None
        if collect_sufficient_stats:
            sufficient_stats = torch.zeros_like(lambda_parameter)

        if exp_e_log_beta is None:
            exp_e_log_beta = self._exp_dirichlet_expectation(lambda_parameter)

        batch_start = 0
        for prepared_batch in prepared_matrix.batches:
            batch_end = batch_start + prepared_batch.n_docs
            batch_gamma, batch_stats = self._infer_gamma_batch(
                prepared_batch=prepared_batch,
                exp_e_log_beta=exp_e_log_beta,
                doc_topic_prior=doc_topic_prior,
                max_doc_update_iter=max_doc_update_iter,
                mean_change_tol=mean_change_tol,
                collect_sufficient_stats=collect_sufficient_stats,
                sufficient_stats_buffer=sufficient_stats,
            )
            gamma[batch_start:batch_end] = batch_gamma
            if collect_sufficient_stats and batch_stats is not sufficient_stats:
                sufficient_stats += batch_stats
            batch_start = batch_end

        return gamma, sufficient_stats

    def _estimate_bound(self, prepared_matrix, document_topic_matrix, lambda_parameter):
        beta = self._normalize_rows(lambda_parameter)
        total = torch.tensor(0.0, device=self.device, dtype=self._torch_dtype())
        batch_start = 0

        for prepared_batch in prepared_matrix.batches:
            batch_end = batch_start + prepared_batch.n_docs
            rows = prepared_batch.rows
            cols = prepared_batch.cols
            counts = prepared_batch.counts
            if rows.numel() == 0:
                batch_start = batch_end
                continue

            doc_topic_batch = document_topic_matrix[batch_start:batch_end]
            doc_topic_rows = doc_topic_batch.index_select(0, rows)
            beta_cols = beta.index_select(1, cols).transpose(0, 1)
            word_probabilities = (doc_topic_rows * beta_cols).sum(dim=1).clamp_min(1e-12)
            total = total + (counts * torch.log(word_probabilities)).sum()
            batch_start = batch_end

        return float(total.detach().cpu().item())

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
        return_document_topic_matrix = bool(kwargs.pop("return_document_topic_matrix", True))

        if learning_method not in {"batch", "online"}:
            raise ValueError("learning_method must be 'batch' or 'online'")

        prepared_matrix = self._prepare_sparse_batches(data_matrix, batch_size=batch_size)
        n_components = int(n_components)
        n_features = prepared_matrix.n_features
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
            exp_e_log_beta = self._exp_dirichlet_expectation(lambda_parameter)
            gamma, sufficient_stats = self._infer_gamma(
                prepared_matrix=prepared_matrix,
                lambda_parameter=lambda_parameter,
                exp_e_log_beta=exp_e_log_beta,
                doc_topic_prior=doc_topic_prior,
                max_doc_update_iter=max_doc_update_iter,
                mean_change_tol=mean_change_tol,
                collect_sufficient_stats=True,
            )
            lambda_parameter = sufficient_stats + float(topic_word_prior)

        exp_dirichlet_component = self._exp_dirichlet_expectation(lambda_parameter)
        document_topic_matrix = self._normalize_rows(gamma)
        bound = self._estimate_bound(prepared_matrix, document_topic_matrix, lambda_parameter)

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

        document_topic_matrix_output = None
        if return_document_topic_matrix:
            document_topic_matrix_output = document_topic_matrix.detach().cpu().numpy()

        return self._fit_result_type()(model=model, document_topic_matrix=document_topic_matrix_output)

    def transform(self, model, data_matrix):
        prepared_matrix = self._prepare_sparse_batches(
            data_matrix,
            batch_size=self.options.get("batch_size", 256),
        )
        gamma, _ = self._infer_gamma(
            prepared_matrix=prepared_matrix,
            lambda_parameter=model.lambda_parameter,
            exp_e_log_beta=model.exp_dirichlet_component_tensor,
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