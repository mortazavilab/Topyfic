import argparse
import gc
import json
import statistics
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from Topyfic.backends.torch_backend import torch
from Topyfic.train import Train


def topic_alignment_cost(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left / left.sum(axis=1, keepdims=True)
    right = right / right.sum(axis=1, keepdims=True)
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=1, keepdims=True) * np.linalg.norm(right, axis=1, keepdims=True).T
    similarity = numerator / denominator
    rows, cols = linear_sum_assignment(1 - similarity)
    order = np.argsort(rows)
    rows = rows[order]
    cols = cols[order]
    return rows, cols, 1 - similarity[rows, cols]


def summarize_durations(samples, warmup_runs=1):
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if len(samples) <= warmup_runs:
        raise ValueError("samples must include at least one measured run")

    numeric_samples = [float(sample) for sample in samples]
    measured = numeric_samples[warmup_runs:]
    return {
        "all_seconds": numeric_samples,
        "warmup_seconds": numeric_samples[:warmup_runs],
        "measured_seconds": measured,
        "mean_seconds": statistics.mean(measured),
        "median_seconds": statistics.median(measured),
        "stdev_seconds": statistics.pstdev(measured),
        "min_seconds": min(measured),
        "max_seconds": max(measured),
    }


def resolve_requested_devices(requested_devices):
    available = []
    unavailable = {}

    for raw_device in requested_devices:
        device = raw_device.strip().lower()
        if device == "":
            continue
        if device == "cpu":
            available.append(device)
            continue
        if device == "mps":
            if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                available.append(device)
            else:
                unavailable[device] = "MPS is not available"
            continue
        if device == "cuda":
            if torch is not None and torch.cuda.is_available():
                available.append(device)
            else:
                unavailable[device] = "CUDA is not available"
            continue
        raise ValueError(f"Unsupported device '{raw_device}'")

    return available, unavailable


def resolve_parity_reference_device(available_devices):
    normalized_devices = [device.strip().lower() for device in available_devices if device.strip() != ""]
    if "cpu" in normalized_devices:
        return "cpu", [device for device in normalized_devices if device != "cpu"]
    if normalized_devices:
        return normalized_devices[0], normalized_devices[1:]
    return None, []


def _synchronize_device(device):
    if torch is None:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def _clear_device_cache(device):
    if torch is None:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _time_callable(func, device):
    _synchronize_device(device)
    start = time.perf_counter()
    value = func()
    _synchronize_device(device)
    elapsed = time.perf_counter() - start
    return value, elapsed


def _make_train(adata, backend_name, backend_kwargs, k, random_state, batch_size, max_iter):
    train = Train(
        name=f"benchmark_{backend_name}_{backend_kwargs.get('device', 'cpu')}",
        k=k,
        n_runs=1,
        random_state_range=[random_state],
        backend_name=backend_name,
        backend_kwargs=backend_kwargs,
    )
    _, train_seconds = _time_callable(
        lambda: train.run_LDA_models(
            adata,
            batch_size=batch_size,
            max_iter=max_iter,
            n_jobs=1,
            n_thread=1,
        ),
        backend_kwargs.get("device"),
    )
    _, first_transform_seconds = _time_callable(
        lambda: train.top_models[0].transform(adata.X),
        backend_kwargs.get("device"),
    )
    _, second_transform_seconds = _time_callable(
        lambda: train.top_models[0].transform(adata.X),
        backend_kwargs.get("device"),
    )
    return train, train_seconds, first_transform_seconds, second_transform_seconds


def benchmark_backends(adata,
                       *,
                       k=5,
                       random_state=0,
                       batch_size=256,
                       max_iter=5,
                       dtype="float32",
                       devices=("cpu", "mps", "cuda"),
                       repeats=3,
                       warmup_runs=1):
    total_runs = int(repeats) + int(warmup_runs)
    if total_runs < 1:
        raise ValueError("repeats + warmup_runs must be at least 1")

    requested_devices, unavailable_devices = resolve_requested_devices(devices)
    configs = [("sklearn", "sklearn", {})]
    configs.extend(
        (f"torch_{device}", "torch", {"device": device, "dtype": dtype})
        for device in requested_devices
    )

    results = []
    for label, backend_name, backend_kwargs in configs:
        train_samples = []
        first_transform_samples = []
        second_transform_samples = []

        for _ in range(total_runs):
            gc.collect()
            _clear_device_cache(backend_kwargs.get("device"))
            _, train_seconds, first_transform_seconds, second_transform_seconds = _make_train(
                adata=adata,
                backend_name=backend_name,
                backend_kwargs=backend_kwargs,
                k=k,
                random_state=random_state,
                batch_size=batch_size,
                max_iter=max_iter,
            )
            train_samples.append(train_seconds)
            first_transform_samples.append(first_transform_seconds)
            second_transform_samples.append(second_transform_seconds)

        results.append(
            {
                "label": label,
                "backend": backend_name,
                "backend_kwargs": backend_kwargs,
                "train": summarize_durations(train_samples, warmup_runs=warmup_runs),
                "first_transform": summarize_durations(first_transform_samples, warmup_runs=warmup_runs),
                "second_transform": summarize_durations(second_transform_samples, warmup_runs=warmup_runs),
            }
        )

    baseline = next(item["train"]["mean_seconds"] for item in results if item["label"] == "sklearn")
    for item in results:
        item["train_speedup_vs_sklearn_mean"] = baseline / item["train"]["mean_seconds"]

    return {
        "dataset_shape": [int(adata.n_obs), int(adata.n_vars)],
        "requested_devices": [device for device in devices],
        "unavailable_devices": unavailable_devices,
        "results": results,
    }


def benchmark_parity(adata,
                     *,
                     k=5,
                     random_state=0,
                     batch_size=256,
                     max_iter=5,
                     dtype="float32",
                     devices=("cpu", "mps", "cuda")):
    requested_devices, unavailable_devices = resolve_requested_devices(devices)
    reference_device, candidate_devices = resolve_parity_reference_device(requested_devices)
    if reference_device is None:
        return {
            "dataset_shape": [int(adata.n_obs), int(adata.n_vars)],
            "requested_devices": [device for device in devices],
            "unavailable_devices": unavailable_devices,
            "reference_backend": None,
            "results": [],
        }

    reference_train, _, _, _ = _make_train(
        adata=adata,
        backend_name="torch",
        backend_kwargs={"device": reference_device, "dtype": dtype},
        k=k,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
    )
    reference_model = reference_train.top_models[0]
    reference_components = reference_model.model.components_
    reference_transform = reference_model.transform(adata.X)

    results = []
    for device in candidate_devices:
        candidate_train, _, _, _ = _make_train(
            adata=adata,
            backend_name="torch",
            backend_kwargs={"device": device, "dtype": dtype},
            k=k,
            random_state=random_state,
            batch_size=batch_size,
            max_iter=max_iter,
        )
        candidate_model = candidate_train.top_models[0]
        _, alignment, costs = topic_alignment_cost(reference_components, candidate_model.model.components_)
        aligned_components = candidate_model.model.components_[alignment]
        aligned_transform = candidate_model.transform(adata.X)[:, alignment]
        results.append(
            {
                "device": device,
                "alignment": alignment.tolist(),
                "max_component_cosine_distance": float(np.max(costs)),
                "max_component_abs_diff": float(np.max(np.abs(aligned_components - reference_components))),
                "max_cell_participation_abs_diff": float(np.max(np.abs(aligned_transform - reference_transform))),
            }
        )

    return {
        "dataset_shape": [int(adata.n_obs), int(adata.n_vars)],
        "requested_devices": [device for device in devices],
        "unavailable_devices": unavailable_devices,
        "reference_backend": {"backend": "torch", "device": reference_device, "dtype": dtype},
        "results": results,
    }


def build_benchmark_report(adata,
                           *,
                           k=5,
                           random_state=0,
                           batch_size=256,
                           max_iter=5,
                           dtype="float32",
                           devices=("cpu", "mps", "cuda"),
                           repeats=3,
                           warmup_runs=1):
    return {
        "timing": benchmark_backends(
            adata,
            k=k,
            random_state=random_state,
            batch_size=batch_size,
            max_iter=max_iter,
            dtype=dtype,
            devices=devices,
            repeats=repeats,
            warmup_runs=warmup_runs,
        ),
        "parity": benchmark_parity(
            adata,
            k=k,
            random_state=random_state,
            batch_size=batch_size,
            max_iter=max_iter,
            dtype=dtype,
            devices=devices,
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark sklearn and torch LDA backends across CPU, MPS, and CUDA")
    parser.add_argument("adata", help="Path to an AnnData .h5ad file")
    parser.add_argument("--k", type=int, default=5, help="Number of topics")
    parser.add_argument("--random-state", type=int, default=0, help="Random state for all runs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--max-iter", type=int, default=5, help="Maximum LDA iterations")
    parser.add_argument("--dtype", default="float32", help="Torch dtype for torch backends")
    parser.add_argument(
        "--devices",
        default="cpu,mps,cuda",
        help="Comma-separated torch devices to benchmark alongside sklearn",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Measured timing runs per backend")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed warmup-style runs per backend")
    parser.add_argument("--output", help="Optional path to write the JSON report")
    args = parser.parse_args(argv)

    try:
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError("scanpy is required to load AnnData benchmark inputs") from exc

    adata = sc.read_h5ad(args.adata)
    report = build_benchmark_report(
        adata,
        k=args.k,
        random_state=args.random_state,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        dtype=args.dtype,
        devices=tuple(args.devices.split(",")),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
    )
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    print(payload)


if __name__ == "__main__":
    main()