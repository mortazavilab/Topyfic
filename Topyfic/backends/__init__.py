from sklearn.decomposition import LatentDirichletAllocation

from Topyfic.backends.base import LDABackend, LDAFitResult
from Topyfic.backends.sklearn_backend import SklearnLDABackend
from Topyfic.backends.torch_backend import TorchLDABackend, TorchLDAModel


BACKEND_REGISTRY = {
    "sklearn": SklearnLDABackend,
    "torch": TorchLDABackend,
}


def default_lda_backend_name():
    if TorchLDABackend.is_available():
        return "torch"
    return "sklearn"


def resolve_lda_backend_name(name=None):
    if name in {None, "", "default"}:
        return default_lda_backend_name()
    return name


def create_lda_backend(name=None, **options):
    name = resolve_lda_backend_name(name)
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
    "default_lda_backend_name",
    "infer_backend_name",
    "resolve_lda_backend_name",
]