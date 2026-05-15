from pathlib import Path
import importlib

import numpy as np
import anndata

from Topyfic.analysis import Analysis
from Topyfic.lda_state import LDAState
from Topyfic.train import Train
from Topyfic.utilsMakeModel import _neighbors_kwargs_for_adata, read_topModel, read_train


def test_make_single_lda_model_produces_expected_shapes(synthetic_adata):
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[7])

    top_model = train.make_single_LDA_model(
        synthetic_adata,
        random_state=7,
        name=train.name,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        kwargs={},
    )

    assert top_model.name == "demo_7"
    assert top_model.N == 2
    assert top_model.model.components_.shape == (2, synthetic_adata.n_vars)
    assert top_model.get_gene_weights().shape == (synthetic_adata.n_vars, 2)
    assert top_model.get_feature_name() == synthetic_adata.var_names.tolist()


def test_run_lda_models_single_thread_avoids_pool(monkeypatch, synthetic_adata):
    train_module = importlib.import_module("Topyfic.train")
    train = Train(name="demo", k=2, n_runs=2, random_state_range=[0, 1])
    recorded_states = []

    def fake_make_single(self, data, random_state, name, learning_method, batch_size, max_iter, n_jobs, kwargs):
        recorded_states.append(random_state)
        return f"model-{random_state}"

    class FailIfInstantiated:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Pool should not be instantiated for n_thread=1")

    monkeypatch.setattr(train_module, "Pool", FailIfInstantiated)
    monkeypatch.setattr(Train, "make_single_LDA_model", fake_make_single)

    train.run_LDA_models(synthetic_adata, n_thread=1)

    assert train.top_models == ["model-0", "model-1"]
    assert recorded_states == [0, 1]


def test_make_lda_model_attributes_match_run_count_and_features(synthetic_adata):
    train = Train(name="demo", k=2, n_runs=2, random_state_range=[0, 1])
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    all_components, all_exp_dirichlet_component, all_others = train.make_LDA_models_attributes()

    assert all_components.shape == (train.n_runs * train.k, synthetic_adata.n_vars)
    assert all_exp_dirichlet_component.shape == (train.n_runs * train.k, synthetic_adata.n_vars)
    assert all_others.shape == (train.n_runs * train.k, 6)
    assert all_components.index.tolist() == ["Topic1_R0", "Topic2_R0", "Topic1_R1", "Topic2_R1"]


def test_lda_state_round_trip_preserves_sklearn_contract(synthetic_adata):
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[0])
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    top_model = train.top_models[0]
    state = LDAState.from_sklearn_model(
        top_model.model,
        feature_names=top_model.get_feature_name(),
        topic_names=[f"Topic_{index + 1}" for index in range(top_model.N)],
    )
    rebuilt_model = state.to_sklearn_model()

    np.testing.assert_allclose(rebuilt_model.components_, top_model.model.components_)
    np.testing.assert_allclose(rebuilt_model.exp_dirichlet_component_, top_model.model.exp_dirichlet_component_)
    assert rebuilt_model.n_batch_iter_ == top_model.model.n_batch_iter_
    assert rebuilt_model.n_features_in_ == top_model.model.n_features_in_
    assert rebuilt_model.n_iter_ == top_model.model.n_iter_
    assert rebuilt_model.bound_ == top_model.model.bound_
    assert rebuilt_model.doc_topic_prior_ == top_model.model.doc_topic_prior_
    assert rebuilt_model.topic_word_prior_ == top_model.model.topic_word_prior_


def test_train_pickle_round_trip(tmp_path, synthetic_adata):
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[0])
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    train.save_train(save_path=f"{tmp_path}/")
    reloaded = read_train(f"{tmp_path}/train_demo.p")

    assert reloaded.name == train.name
    assert reloaded.k == train.k
    assert len(reloaded.top_models) == len(train.top_models)
    np.testing.assert_allclose(
        reloaded.top_models[0].model.components_,
        train.top_models[0].model.components_,
    )


def test_train_hdf5_round_trip(tmp_path, synthetic_adata):
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[0])
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    train.save_train(save_path=f"{tmp_path}/", file_format="HDF5")

    output_file = tmp_path / "train_demo.h5"
    assert output_file.exists()

    reloaded = read_train(str(output_file))

    assert reloaded.name == train.name
    assert reloaded.k == train.k
    assert reloaded.backend_name == "sklearn"
    assert len(reloaded.top_models) == len(train.top_models)
    assert reloaded.top_models[0].get_feature_name() == synthetic_adata.var_names.tolist()
    np.testing.assert_allclose(
        reloaded.top_models[0].model.components_,
        train.top_models[0].model.components_,
    )


def test_topmodel_hdf5_round_trip(tmp_path, synthetic_adata):
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[0])
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    top_model = train.top_models[0]
    top_model.save_topModel(save_path=f"{tmp_path}/", file_format="HDF5")

    output_file = tmp_path / f"topModel_{top_model.name}.h5"
    assert output_file.exists()

    reloaded = read_topModel(str(output_file))

    assert reloaded.name == top_model.name
    assert reloaded.N == top_model.N
    assert reloaded.backend_name == "sklearn"
    assert reloaded.get_feature_name() == top_model.get_feature_name()
    np.testing.assert_allclose(reloaded.model.components_, top_model.model.components_)


def test_analysis_cell_participation_matches_input_shape(synthetic_adata):
    train = Train(name="demo", k=2, n_runs=1, random_state_range=[0])
    train.run_LDA_models(
        synthetic_adata,
        learning_method="batch",
        batch_size=2,
        max_iter=5,
        n_jobs=1,
        n_thread=1,
    )

    analysis = Analysis(Top_model=train.top_models[0])
    analysis.calculate_cell_participation(synthetic_adata)

    assert analysis.cell_participation.shape == (synthetic_adata.n_obs, train.k)
    assert analysis.cell_participation.obs_names.tolist() == synthetic_adata.obs_names.tolist()
    assert analysis.cell_participation.var_names.tolist() == ["Topic_1", "Topic_2"]


def test_neighbors_kwargs_shrink_pca_for_small_topic_tables():
    adata = anndata.AnnData(np.ones((10, 100)))

    assert _neighbors_kwargs_for_adata(adata) == {"n_pcs": 9}