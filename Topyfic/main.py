from __future__ import annotations

import pandas as pd
import scanpy as sc
import click

from Topyfic.backends import default_lda_backend_name, resolve_lda_backend_name
from Topyfic.utilsMakeModel import make_analysis_class, make_topModel, read_topModel, read_train, train_model


@click.group()
def cli():
    pass


def _parse_random_state_range(random_states, random_state_range, n_runs):
    if random_states:
        return list(random_states)

    if random_state_range in (None, "", "range(n_run)", "range(n_runs)"):
        return list(range(n_runs))

    if random_state_range.startswith("range(") and random_state_range.endswith(")"):
        stop = random_state_range[6:-1].strip()
        if stop in {"n_run", "n_runs"}:
            return list(range(n_runs))
        return list(range(int(stop)))

    return [int(value.strip()) for value in random_state_range.split(",") if value.strip()]


def _load_adata(data_path):
    return sc.read_h5ad(data_path)


def _split_paths(raw_values):
    paths = []
    for raw_value in raw_values:
        paths.extend(value.strip() for value in raw_value.split(",") if value.strip())
    return paths


def _load_colors_topics(table_path):
    if table_path is None:
        return None

    separator = "\t" if table_path.endswith(".tsv") else ","
    return pd.read_csv(table_path, sep=separator, index_col=0)


#### train model ####
@cli.command(name='train_model')
@click.option('--name',
              help='name of the Train class',
              required=True)
@click.option('--data',
              help='expression data embedded in anndata format along with cells and genes/region information',
              required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('-k',
              help='number of topics to learn one LDA model using sklearn package',
              required=True,
              type=int)
@click.option('--n-runs', '--n_runs', 'n_runs',
              help='number of run to define rLDA model',
              default=100,
              show_default=True,
              type=int)
@click.option('--random-state', 'random_states',
              help='repeat to provide explicit random states',
              multiple=True,
              type=int)
@click.option('--random-state-range', '--random_state_range', 'random_state_range',
              help='comma-separated random states or range(...) expression',
              default="range(n_runs)",
              show_default=True)
@click.option('--n-thread', '--n_thread', 'n_thread',
              help='',
              default=5,
              show_default=True,
              type=int)
@click.option('--backend', 'backend_name',
              help='internal LDA backend used for training',
              type=click.Choice(['default', 'sklearn', 'torch']),
              default='default',
              show_default=True)
@click.option('--device',
              help='device passed to the torch backend',
              default='auto',
              show_default=True)
@click.option('--dtype',
              help='floating point precision passed to the torch backend',
              type=click.Choice(['float32', 'float64']),
              default='float32',
              show_default=True)
@click.option('--learning-method',
              help='method used to update topic weights',
              type=click.Choice(['online', 'batch']),
              default='online',
              show_default=True)
@click.option('--batch-size',
              help='number of documents to use in each training iteration',
              default=1000,
              show_default=True,
              type=int)
@click.option('--max-iter',
              help='maximum number of training iterations',
              default=10,
              show_default=True,
              type=int)
@click.option('--n-jobs',
              help='parallel jobs for backends that support it',
              default=None,
              type=int)
@click.option('--save-path', '--save_path', 'save_path',
              help='',
              default="",
              show_default=True)
def train_model_command(name,
                        data,
                        k,
                        n_runs=100,
                        random_states=(),
                        random_state_range=None,
                        n_thread=5,
                        backend_name='default',
                        device='auto',
                        dtype='float32',
                        learning_method='online',
                        batch_size=1000,
                        max_iter=10,
                        n_jobs=None,
                        save_path=""):
    backend_name = resolve_lda_backend_name(backend_name)
    backend_kwargs = {}
    if backend_name == 'torch':
        backend_kwargs = {'device': device, 'dtype': dtype}

    return train_model(name=name,
                       data=_load_adata(data),
                       k=k,
                       n_runs=n_runs,
                       random_state_range=_parse_random_state_range(random_states, random_state_range, n_runs),
                       n_thread=n_thread,
                       learning_method=learning_method,
                       batch_size=batch_size,
                       max_iter=max_iter,
                       n_jobs=n_jobs,
                       backend_name=backend_name,
                       backend_kwargs=backend_kwargs,
                       save_path=save_path)


#### top model ####
@cli.command(name='make_topModel')
@click.option('--train-file', '--trains', 'train_files',
              help='list of train class',
              required=True,
              multiple=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--data',
              help='expression data embedded in anndata format along with cells and genes/region information',
              required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--n-top-genes', '--n_top_genes', 'n_top_genes',
              help='Number of highly-variable genes to keep',
              default=50,
              show_default=True,
              type=int)
@click.option('--resolution',
              help='A parameter value controlling the coarseness of the clustering. Higher values lead to more clusters.',
              default=1,
              show_default=True,
              type=float)
@click.option('--file-format', '--file_format', 'file_format',
              help='indicate the format of plot',
              default="pdf",
              show_default=True)
@click.option('--save-path', '--save_path', 'save_path',
              help='directory you want to use to save pickle file',
              default="",
              show_default=True)
def make_topModel_command(train_files,
                          data,
                          n_top_genes=50,
                          resolution=1,
                          file_format="pdf",
                          save_path=""):
    trains = [read_train(path) for path in _split_paths(train_files)]
    return make_topModel(trains=trains,
                         data=_load_adata(data),
                         n_top_genes=n_top_genes,
                         resolution=resolution,
                         file_format=file_format,
                         save_path=save_path)


#### top model ####
@cli.command(name='make_analysis_class')
@click.option('--top-model', '--top_model', 'top_model_file',
              help='top model',
              required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--data',
              help='expression data embedded in anndata format along with cells and genes/region information',
              required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--colors-topics', '--colors_topics', 'colors_topics',
              help='dataframe that mapped colored to topics',
              default=None,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--save-path', '--save_path', 'save_path',
              help='directory you want to use to save pickle file',
              default="",
              show_default=True)
def make_analysis_class_command(top_model_file,
                                data,
                                colors_topics=None,
                                save_path=""):
    return make_analysis_class(top_model=read_topModel(top_model_file),
                               data=_load_adata(data),
                               colors_topics=_load_colors_topics(colors_topics),
                               save_path=save_path)


if __name__ == '__main__':
    cli()
