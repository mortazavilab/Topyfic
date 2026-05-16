# Topyfic Nextflow workflow

This directory contains the Nextflow DSL2 replacement for the legacy Snakemake pipeline.

The pipeline keeps the same major stages:

- train single-run `Train` objects for each random seed
- combine them into a per-dataset `Train`
- build a `TopModel`
- build the matching `Analysis`
- optionally merge multiple datasets into a shared model

## Getting started

Run the pipeline from the repository root so relative paths in the example params file stay predictable.

### 1. Prepare an environment

Use a Python environment with Topyfic and its runtime dependencies installed. For local development, the repository venv works:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .[dev]
```

### 2. Prepare a params file

Copy [params.example.yml](params.example.yml) and replace the placeholder AnnData paths and output directory.

For the checked-in IGVF smoke-test params file, you can deterministically rebuild the expected 1,000-cell subset with:

```bash
python workflow/nextflow/bin/prepare_igvf_subset.py
```

By default the helper downloads the public IGVF matrix file, samples 1,000 cells with seed `0`, and writes [tutorials/IGVFFI3320ZCCE/IGVFFI3320ZCCE_subset_1000.h5ad](tutorials/IGVFFI3320ZCCE/IGVFFI3320ZCCE_subset_1000.h5ad). Use `--force` to overwrite an existing subset.

Key params:

- `names`: dataset identifiers used in output file names
- `count_adata`: map from dataset name to input `.h5ad`
- `n_topics`: initial topic counts to evaluate
- `train.backend`: backend used for each single-run training job (`default` resolves to torch when available, otherwise sklearn)
- `train.device`: target torch device (`auto`, `cpu`, `cuda`, or `mps`)
- `train.dtype`: torch floating point precision passed to the backend
- `train.random_states`: random seeds for single-run training
- `top_model.*`: clustering and filtering settings for `calculate_leiden_clustering`
- `plotting.interactive`: when `false` (default), the pipeline forces a non-GUI Matplotlib backend so plots are saved without opening interactive windows
- `merge`: whether to build a merged model across all datasets

### 3. Validate the DAG

Use Nextflow's stub mode first. It validates channel wiring and output contracts without running the Python jobs.

```bash
nextflow run workflow/nextflow/main.nf \
  -params-file workflow/nextflow/params.example.yml \
  -stub-run
```

### 4. Run the workflow

```bash
nextflow run workflow/nextflow/main.nf \
  -params-file workflow/nextflow/params.example.yml
```

To allow interactive plotting for local exploratory runs, set `plotting.interactive: true` in your params file.

By default, the workflow uses `train.backend: default`, which resolves to torch when PyTorch is installed and otherwise falls back to sklearn. To force a backup path, set `train.backend: torch` with an explicit `train.device`, or set `train.backend: sklearn`.

Outputs are written under `workdir` using the same directory structure as the legacy workflow:

- `{workdir}/{name}/{n_topic}/train/`
- `{workdir}/{name}/{n_topic}/topmodel/`
- `{workdir}/` for merged outputs