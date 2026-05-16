import pytest
import numpy as np
from anndata import AnnData
from scipy import sparse as sp

from Topyfic.backends import TorchLDABackend, create_lda_backend, default_lda_backend_name
from Topyfic.benchmarking import topic_alignment_cost
from Topyfic.backends.torch_backend import torch
from Topyfic.train import Train
from Topyfic.utilsMakeModel import combine_topModels, filter_LDA_model, initialize_rLDA_model, read_topModel, read_train


def test_create_lda_backend_returns_sklearn_backend():
    backend = create_lda_backend("sklearn")

    assert backend.name == "sklearn"


def test_create_lda_backend_rejects_unknown_backend():
    with pytest.raises(ValueError):
        create_lda_backend("does-not-exist")


def test_train_defaults_to_available_backend(synthetic_adata):
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[0])
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    expected_backend = default_lda_backend_name()
    assert train.backend_name == expected_backend
    assert train.top_models[0].backend_name == expected_backend


def test_backend_property_reuses_backend_instance(synthetic_adata):
    train = Train(
        name="demo",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float32"},
    )

    assert train.backend is train.backend

    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    assert train.top_models[0].backend is train.top_models[0].backend


def test_train_sklearn_backend_accepts_sparse_input(synthetic_adata):
    sparse_adata = AnnData(sp.csr_matrix(synthetic_adata.X), obs=synthetic_adata.obs.copy(), var=synthetic_adata.var.copy())
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[0])
    train.run_LDA_models(
        sparse_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    assert train.top_models[0].model.components_.shape == (2, synthetic_adata.n_vars)


def test_torch_backend_resolves_cpu_without_torch():
    assert TorchLDABackend.resolve_device("cpu") == "cpu"


def _run_torch_train_on_device(synthetic_adata, device, dtype="float32", max_iter=10):
    train = Train(
        name=f"demo_{device}",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": device, "dtype": dtype},
    )
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=max_iter,
        n_jobs=1,
        n_thread=1,
    )
    return train


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_torch_backend_fit_returns_probability_matrix(synthetic_adata):
    backend = TorchLDABackend(device="cpu", dtype="float64")

    fit_result = backend.fit(
        data_matrix=synthetic_adata.X,
        n_components=2,
        random_state=0,
        learning_method="batch",
        max_iter=15,
    )

    assert fit_result.model.components_.shape == (2, synthetic_adata.n_vars)
    assert fit_result.document_topic_matrix.shape == (synthetic_adata.n_obs, 2)
    np.testing.assert_allclose(fit_result.document_topic_matrix.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(fit_result.model.bound_)


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_torch_backend_state_round_trip_preserves_model(synthetic_adata):
    backend = TorchLDABackend(device="cpu", dtype="float64")

    fit_result = backend.fit(
        data_matrix=synthetic_adata.X,
        n_components=2,
        random_state=0,
        learning_method="batch",
        max_iter=15,
    )
    state = backend.get_state(
        fit_result.model,
        feature_names=synthetic_adata.var_names.tolist(),
        topic_names=["Topic_1", "Topic_2"],
    )
    rebuilt_model = backend.model_from_state(state)

    np.testing.assert_allclose(rebuilt_model.components_, fit_result.model.components_)
    np.testing.assert_allclose(rebuilt_model.exp_dirichlet_component_, fit_result.model.exp_dirichlet_component_)
    assert rebuilt_model.n_iter_ == fit_result.model.n_iter_


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_train_can_use_torch_backend_on_cpu(synthetic_adata):
    train = Train(
        name="demo",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )

    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    assert train.top_models[0].backend_name == "torch"
    transformed = train.top_models[0].transform(synthetic_adata.X)
    assert transformed.shape == (synthetic_adata.n_obs, train.k)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_train_can_use_torch_backend_with_sparse_input(synthetic_adata):
    sparse_adata = AnnData(sp.csr_matrix(synthetic_adata.X), obs=synthetic_adata.obs.copy(), var=synthetic_adata.var.copy())
    train = Train(
        name="demo",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float32"},
    )

    train.run_LDA_models(
        sparse_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=10,
        n_jobs=1,
        n_thread=1,
    )

    transformed = train.top_models[0].transform(sparse_adata.X)
    assert transformed.shape == (sparse_adata.n_obs, train.k)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_torch_train_hdf5_round_trip_preserves_backend_metadata(tmp_path, synthetic_adata):
    train = Train(
        name="demo",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    train.save_train(save_path=f"{tmp_path}/", file_format="HDF5")
    reloaded = read_train(f"{tmp_path}/train_demo.h5")

    assert reloaded.backend_name == "torch"
    assert reloaded.backend_kwargs == {"device": "cpu", "dtype": "float64"}
    assert reloaded.top_models[0].backend_name == "torch"
    assert reloaded.top_models[0].backend_kwargs == {"device": "cpu", "dtype": "float64"}
    assert reloaded.top_models[0].get_feature_name() == synthetic_adata.var_names.tolist()

    transformed = reloaded.top_models[0].transform(synthetic_adata.X)
    assert transformed.shape == (synthetic_adata.n_obs, 2)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_filter_lda_model_preserves_torch_backend(synthetic_adata):
    train = Train(
        name="demo",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    filtered_model, filtered_components = filter_LDA_model(
        train.top_models[0].model,
        np.array([True, False]),
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
        feature_names=synthetic_adata.var_names.tolist(),
    )

    backend = create_lda_backend("torch", device="cpu", dtype="float64")
    transformed = backend.transform(filtered_model, synthetic_adata.X)

    assert getattr(filtered_model, "backend_name", None) == "torch"
    assert filtered_components.shape == (1, synthetic_adata.n_vars)
    assert transformed.shape == (synthetic_adata.n_obs, 1)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_initialize_rlda_model_preserves_torch_backend(synthetic_adata):
    train = Train(
        name="demo",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    all_components, all_exp_dirichlet_component, all_others = train.make_LDA_models_attributes()
    clustering = all_components.copy(deep=True)
    clustering["leiden"] = [0, 1]

    rlda = initialize_rLDA_model(
        all_components,
        all_exp_dirichlet_component,
        all_others,
        clusters=clustering,
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )

    backend = create_lda_backend("torch", device="cpu", dtype="float64")
    transformed = backend.transform(rlda, synthetic_adata.X)

    assert getattr(rlda, "backend_name", None) == "torch"
    assert transformed.shape == (synthetic_adata.n_obs, 2)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_torch_topmodel_hdf5_round_trip_preserves_backend_metadata(tmp_path, synthetic_adata):
    train = Train(
        name="demo",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    top_model = train.top_models[0]
    top_model.save_topModel(save_path=f"{tmp_path}/", file_format="HDF5")
    reloaded = read_topModel(f"{tmp_path}/topModel_{top_model.name}.h5")

    assert reloaded.backend_name == "torch"
    assert reloaded.backend_kwargs == {"device": "cpu", "dtype": "float64"}
    assert reloaded.get_feature_name() == synthetic_adata.var_names.tolist()

    transformed = reloaded.transform(synthetic_adata.X)
    assert transformed.shape == (synthetic_adata.n_obs, 2)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_combine_topmodels_preserves_torch_backend(synthetic_adata):
    first = Train(
        name="first",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )
    first.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    second = Train(
        name="second",
        k=2,
        n_runs=1,
        random_state_range=[1],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )
    second.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    combined, n_topics, gene_weights = combine_topModels(
        [first.top_models[0], second.top_models[0]],
        name="combined",
        data=synthetic_adata,
        min_cell_participation=0,
    )

    transformed = combined.transform(synthetic_adata.X)

    assert combined.backend_name == "torch"
    assert combined.backend_kwargs == {"device": "cpu", "dtype": "float64"}
    assert n_topics == combined.N
    assert gene_weights.shape[1] == combined.N
    assert transformed.shape == (synthetic_adata.n_obs, combined.N)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not TorchLDABackend.is_available(), reason="torch is not installed")
def test_torch_backend_topics_track_sklearn_on_synthetic_data(synthetic_adata):
    sklearn_train = Train(name="sk", k=2, n_runs=1, random_state_range=[0])
    sklearn_train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    torch_train = Train(
        name="torch",
        k=2,
        n_runs=1,
        random_state_range=[0],
        backend_name="torch",
        backend_kwargs={"device": "cpu", "dtype": "float64"},
    )
    torch_train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=15,
        n_jobs=1,
        n_thread=1,
    )

    _, alignment, costs = topic_alignment_cost(
        sklearn_train.top_models[0].model.components_,
        torch_train.top_models[0].model.components_,
    )

    assert costs.max() < 0.1

    sklearn_output = sklearn_train.top_models[0].transform(synthetic_adata.X)
    torch_output = torch_train.top_models[0].transform(synthetic_adata.X)[:, alignment]
    np.testing.assert_allclose(torch_output, sklearn_output, atol=0.2)


@pytest.mark.skipif(
    not TorchLDABackend.is_available() or not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
def test_train_can_use_torch_backend_on_mps(synthetic_adata):
    train = _run_torch_train_on_device(synthetic_adata, device="mps")

    transformed = train.top_models[0].transform(synthetic_adata.X)
    assert transformed.shape == (synthetic_adata.n_obs, train.k)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.skipif(
    not TorchLDABackend.is_available() or not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_train_can_use_torch_backend_on_cuda(synthetic_adata):
    train = _run_torch_train_on_device(synthetic_adata, device="cuda")

    transformed = train.top_models[0].transform(synthetic_adata.X)
    assert transformed.shape == (synthetic_adata.n_obs, train.k)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.skipif(
    not TorchLDABackend.is_available() or not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
def test_torch_backend_mps_matches_cpu(synthetic_adata):
    cpu_train = _run_torch_train_on_device(synthetic_adata, device="cpu")
    mps_train = _run_torch_train_on_device(synthetic_adata, device="mps")

    _, alignment, costs = topic_alignment_cost(
        cpu_train.top_models[0].model.components_,
        mps_train.top_models[0].model.components_,
    )

    assert costs.max() < 1e-4

    cpu_output = cpu_train.top_models[0].transform(synthetic_adata.X)
    mps_output = mps_train.top_models[0].transform(synthetic_adata.X)[:, alignment]
    np.testing.assert_allclose(mps_output, cpu_output, atol=1e-4)


@pytest.mark.skipif(
    not TorchLDABackend.is_available() or not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_torch_backend_cuda_matches_cpu(synthetic_adata):
    cpu_train = _run_torch_train_on_device(synthetic_adata, device="cpu")
    cuda_train = _run_torch_train_on_device(synthetic_adata, device="cuda")

    _, alignment, costs = topic_alignment_cost(
        cpu_train.top_models[0].model.components_,
        cuda_train.top_models[0].model.components_,
    )

    assert costs.max() < 1e-4

    cpu_output = cpu_train.top_models[0].transform(synthetic_adata.X)
    cuda_output = cuda_train.top_models[0].transform(synthetic_adata.X)[:, alignment]
    np.testing.assert_allclose(cuda_output, cpu_output, atol=1e-4)