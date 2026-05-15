from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

from Topyfic.lda_state import LDAState


PERSISTENCE_VERSION = 2
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def _decode_hdf5_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return value


def _replace_text_dataset(handle, name, value):
    if name in handle:
        del handle[name]
    handle.create_dataset(name, data=str(value), dtype=STRING_DTYPE)


def _replace_name_array(handle, name, values):
    if name in handle:
        del handle[name]

    values = list(values)
    if not values:
        handle.create_dataset(name, data=np.array([], dtype=int))
        return

    if all(isinstance(value, str) for value in values):
        handle.create_dataset(name, data=np.asarray(values, dtype=object), dtype=STRING_DTYPE)
        return

    handle.create_dataset(name, data=np.asarray(values))


def _read_text_dataset(handle, name, default=None):
    if name not in handle:
        return default
    return _decode_hdf5_value(handle[name][()])


def _read_name_array(dataset):
    values = dataset[()]
    if np.isscalar(values):
        values = [values]
    return [_decode_hdf5_value(value) for value in values.tolist()]


def write_backend_metadata(handle, backend_name, backend_kwargs=None):
    backend_kwargs = {} if backend_kwargs is None else dict(backend_kwargs)
    backend_kwargs_json = json.dumps(backend_kwargs, sort_keys=True)

    handle.attrs["topyfic_persistence_version"] = int(PERSISTENCE_VERSION)
    handle.attrs["backend_name"] = backend_name
    handle.attrs["backend_kwargs"] = backend_kwargs_json
    _replace_text_dataset(handle, "backend_name", backend_name)
    _replace_text_dataset(handle, "backend_kwargs", backend_kwargs_json)


def read_backend_metadata(handle, default_name="sklearn", default_kwargs=None):
    default_kwargs = {} if default_kwargs is None else dict(default_kwargs)

    backend_name = handle.attrs.get("backend_name")
    if backend_name is None:
        backend_name = _read_text_dataset(handle, "backend_name", default_name)
    backend_name = _decode_hdf5_value(backend_name)

    backend_kwargs_json = handle.attrs.get("backend_kwargs")
    if backend_kwargs_json is None:
        backend_kwargs_json = _read_text_dataset(handle, "backend_kwargs")
    backend_kwargs_json = _decode_hdf5_value(backend_kwargs_json)
    if backend_kwargs_json:
        backend_kwargs = json.loads(backend_kwargs_json)
    else:
        backend_kwargs = default_kwargs

    persistence_version = int(handle.attrs.get("topyfic_persistence_version", 1))

    return backend_name or default_name, backend_kwargs, persistence_version


def write_lda_state(handle, state: LDAState):
    if "state" in handle:
        del handle["state"]

    state_group = handle.create_group("state")
    state_group.attrs["state_format"] = "lda_state_v1"
    state_group.create_dataset("components", data=state.components.values)
    state_group.create_dataset(
        "exp_dirichlet_component",
        data=state.exp_dirichlet_component.values,
    )
    _replace_name_array(state_group, "feature_names", state.components.columns.tolist())
    _replace_name_array(state_group, "topic_names", state.components.index.tolist())

    others_group = state_group.create_group("others")
    others_group.create_dataset("n_batch_iter", data=int(state.n_batch_iter))
    others_group.create_dataset("n_features_in", data=int(state.n_features_in))
    others_group.create_dataset("n_iter", data=int(state.n_iter))
    others_group.create_dataset("bound", data=float(state.bound))
    others_group.create_dataset("doc_topic_prior", data=float(state.doc_topic_prior))
    others_group.create_dataset("topic_word_prior", data=float(state.topic_word_prior))


def read_lda_state(handle):
    if "state" in handle:
        state_group = handle["state"]
        components_values = np.asarray(state_group["components"])
        exp_dirichlet_component_values = np.asarray(state_group["exp_dirichlet_component"])
        feature_names = _read_name_array(state_group["feature_names"])
        topic_names = _read_name_array(state_group["topic_names"])
        others_group = state_group["others"]

        components = pd.DataFrame(
            components_values,
            index=topic_names,
            columns=feature_names,
        )
        exp_dirichlet_component = pd.DataFrame(
            exp_dirichlet_component_values,
            index=topic_names,
            columns=feature_names,
        )
        others = pd.DataFrame(
            {
                "n_batch_iter": [int(others_group["n_batch_iter"][()])],
                "n_features_in": [int(others_group["n_features_in"][()])],
                "n_iter": [int(others_group["n_iter"][()])],
                "bound": [float(others_group["bound"][()])],
                "doc_topic_prior": [float(others_group["doc_topic_prior"][()])],
                "topic_word_prior": [float(others_group["topic_word_prior"][()])],
            }
        )

        return LDAState.from_frames(components, exp_dirichlet_component, others)

    components = pd.DataFrame(np.asarray(handle["components_"]))
    exp_dirichlet_component = pd.DataFrame(np.asarray(handle["exp_dirichlet_component_"]))
    others = pd.DataFrame(
        {
            "n_batch_iter": [int(handle["n_batch_iter_"][()])],
            "n_features_in": [int(handle["n_features_in_"][()])],
            "n_iter": [int(handle["n_iter_"][()])],
            "bound": [float(handle["bound_"][()])],
            "doc_topic_prior": [float(handle["doc_topic_prior_"][()])],
            "topic_word_prior": [float(handle["topic_word_prior_"][()])],
        }
    )
    return LDAState.from_frames(components, exp_dirichlet_component, others)