import anndata as ad
import numpy as np

from Topyfic.datasets import materialize_igvf_subset, select_deterministic_obs_indices, subset_adata_observations


def test_select_deterministic_obs_indices_is_sorted_and_reproducible():
    first = select_deterministic_obs_indices(n_obs=10, subset_size=4, seed=7)
    second = select_deterministic_obs_indices(n_obs=10, subset_size=4, seed=7)

    assert first.tolist() == second.tolist()
    assert first.tolist() == sorted(first.tolist())
    assert len(first) == 4


def test_subset_adata_observations_preserves_expected_rows(synthetic_adata):
    subset = subset_adata_observations(synthetic_adata, subset_size=3, seed=1)

    assert subset.n_obs == 3
    assert subset.n_vars == synthetic_adata.n_vars
    assert subset.obs_names.tolist() == ["cell_1", "cell_2", "cell_4"]
    assert subset.uns["topyfic_subset"]["seed"] == 1
    assert subset.uns["topyfic_subset"]["subset_size"] == 3


def test_materialize_igvf_subset_writes_local_source(tmp_path, synthetic_adata):
    source_path = tmp_path / "source.h5ad"
    output_path = tmp_path / "subset.h5ad"
    synthetic_adata.write_h5ad(source_path)

    materialize_igvf_subset(
        source_path_or_url=str(source_path),
        output_path=output_path,
        subset_size=4,
        seed=2,
        force=True,
    )

    subset = ad.read_h5ad(output_path)

    assert output_path.exists()
    assert subset.n_obs == 4
    assert subset.n_vars == synthetic_adata.n_vars
    assert subset.uns["topyfic_subset"]["seed"] == 2


def test_materialize_igvf_subset_reuses_existing_output(tmp_path, synthetic_adata):
    source_path = tmp_path / "source.h5ad"
    output_path = tmp_path / "subset.h5ad"
    synthetic_adata.write_h5ad(source_path)

    first_subset = subset_adata_observations(synthetic_adata, subset_size=2, seed=0)
    first_subset.write_h5ad(output_path)

    materialize_igvf_subset(
        source_path_or_url=str(source_path),
        output_path=output_path,
        subset_size=4,
        seed=3,
        force=False,
    )

    subset = ad.read_h5ad(output_path)

    assert subset.n_obs == 2
