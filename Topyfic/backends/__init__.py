from sklearn.decomposition import LatentDirichletAllocation

from Topyfic.backends.base import LDABackend, LDAFitResult
from Topyfic.backends.sklearn_backend import SklearnLDABackend
from Topyfic.backends.torch_backend import TorchLDABackend, TorchLDAModel


BACKEND_REGISTRY = {
    "sklearn": SklearnLDABackend,
    "torch": TorchLDABackend,
}


def create_lda_backend(name="sklearn", **options):
    try:
        backend_cls = BACKEND_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(f"Unknown LDA backend '{name}'. Supported backends: {supported}") from exc

    return backend_cls(**options)


def infer_backend_name(model):
    if model is None:
        return "sklearn"
    if isinstance(model, LatentDirichletAllocation):
        return "sklearn"
    if isinstance(model, TorchLDAModel):
        return "torch"

    return getattr(model, "backend_name", "sklearn")


__all__ = [
    "BACKEND_REGISTRY",
    "LDABackend",
    "LDAFitResult",
    "SklearnLDABackend",
    "TorchLDABackend",
    "TorchLDAModel",
    "create_lda_backend",
    "infer_backend_name",
]