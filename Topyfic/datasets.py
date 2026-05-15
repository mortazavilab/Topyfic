from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from urllib.request import urlretrieve

import anndata as ad
import numpy as np


DEFAULT_IGVF_SUBSET_URL = "https://api.data.igvf.org/matrix-files/IGVFFI3320ZCCE/@@download/IGVFFI3320ZCCE.h5ad"


def select_deterministic_obs_indices(n_obs, subset_size=1000, seed=0):
    if subset_size < 1:
        raise ValueError("subset_size must be at least 1")
    if n_obs < subset_size:
        raise ValueError("subset_size can not be larger than the number of observations")

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=subset_size, replace=False))


def subset_adata_observations(adata, subset_size=1000, seed=0):
    indices = select_deterministic_obs_indices(
        n_obs=adata.n_obs,
        subset_size=subset_size,
        seed=seed,
    )
    subset = adata[indices, :].copy()
    subset.uns["topyfic_subset"] = {
        "method": "deterministic_random_sample",
        "seed": int(seed),
        "subset_size": int(subset_size),
        "source_n_obs": int(adata.n_obs),
    }
    return subset


def _is_remote_path(path_or_url):
    scheme = urlparse(str(path_or_url)).scheme
    return scheme in {"http", "https"}


def _materialize_source_path(source_path_or_url):
    if not _is_remote_path(source_path_or_url):
        return Path(source_path_or_url), None

    temp_dir = TemporaryDirectory()
    download_path = Path(temp_dir.name) / Path(urlparse(source_path_or_url).path).name
    urlretrieve(source_path_or_url, download_path)
    return download_path, temp_dir


def materialize_igvf_subset(source_path_or_url,
                            output_path,
                            subset_size=1000,
                            seed=0,
                            force=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        return output_path

    source_path, temp_dir = _materialize_source_path(source_path_or_url)
    try:
        adata = ad.read_h5ad(source_path)
        subset = subset_adata_observations(adata, subset_size=subset_size, seed=seed)
        subset.write_h5ad(output_path)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    return output_path