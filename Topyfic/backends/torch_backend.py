from __future__ import annotations

from dataclasses import dataclass

from Topyfic.backends.base import LDABackend
from Topyfic.lda_state import LDAState

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent
    torch = None


@dataclass
class TorchLDAModel:
    components: "torch.Tensor"
    exp_dirichlet_component: "torch.Tensor"
    state: LDAState
    device: str
    dtype: str
    backend_name: str = "torch"

    def to_state(self):
        components = self.components.detach().cpu().numpy()
        exp_dirichlet_component = self.exp_dirichlet_component.detach().cpu().numpy()

        return LDAState(
            components=self.state.components.__class__(components, index=self.state.components.index, columns=self.state.components.columns),
            exp_dirichlet_component=self.state.exp_dirichlet_component.__class__(
                exp_dirichlet_component,
                index=self.state.exp_dirichlet_component.index,
                columns=self.state.exp_dirichlet_component.columns,
            ),
            n_batch_iter=self.state.n_batch_iter,
            n_features_in=self.state.n_features_in,
            n_iter=self.state.n_iter,
            bound=self.state.bound,
            doc_topic_prior=self.state.doc_topic_prior,
            topic_word_prior=self.state.topic_word_prior,
        )


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
        raise NotImplementedError(
            "TorchLDABackend.fit is not implemented yet; use the sklearn backend for training until the strict PyTorch solver lands."
        )

    def transform(self, model, data_matrix):
        raise NotImplementedError(
            "TorchLDABackend.transform is not implemented yet; inference will land with the strict PyTorch solver."
        )

    def get_state(self, model, feature_names=None, topic_names=None):
        if isinstance(model, TorchLDAModel):
            return model.to_state()
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
            components=components,
            exp_dirichlet_component=exp_dirichlet_component,
            state=state,
            device=self.device,
            dtype=self.dtype,
        )