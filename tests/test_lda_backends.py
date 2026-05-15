import pytest

from Topyfic.backends import TorchLDABackend, create_lda_backend
from Topyfic.train import Train


def test_create_lda_backend_returns_sklearn_backend():
    backend = create_lda_backend("sklearn")

    assert backend.name == "sklearn"


def test_create_lda_backend_rejects_unknown_backend():
    with pytest.raises(ValueError):
        create_lda_backend("does-not-exist")


def test_train_defaults_to_sklearn_backend(synthetic_adata):
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[0])
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    assert train.backend_name == "sklearn"
    assert train.top_models[0].backend_name == "sklearn"


def test_torch_backend_resolves_cpu_without_torch():
    assert TorchLDABackend.resolve_device("cpu") == "cpu"


def test_torch_backend_fit_is_placeholder():
    backend = TorchLDABackend(device="cpu")

    with pytest.raises(NotImplementedError):
        backend.fit(data_matrix=None, n_components=2, random_state=0)