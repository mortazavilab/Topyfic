from pathlib import Path

import anndata as ad
import scanpy as sc


def ensure_output_dir(path):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_path_arg(path):
    return f"{Path(path).resolve().as_posix()}/"


def load_adata_inputs(adata_paths):
    adata_paths = [Path(adata_path) for adata_path in adata_paths]
    if not adata_paths:
        raise ValueError("At least one AnnData path is required")

    adatas = [sc.read_h5ad(path.as_posix()) for path in adata_paths]
    if len(adatas) == 1:
        return adatas[0]

    return ad.concat(adatas, join='inner', merge='same', label='batch', index_unique='-')


def optional_int(value):
    if value in (None, '', 'None'):
        return None
    return int(value)


def optional_float(value):
    if value in (None, '', 'None'):
        return None
    return float(value)