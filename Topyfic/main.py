import click
from Topyfic.utils import *


@click.group()
def cli():
    pass


#### train model ####
@cli.command(name='train_model')
@click.option('--name',
              help='name of the Train class',
              required=True)
@click.option('--data',
              help='expression data embedded in anndata format along with cells and genes/region information',
              required=True)
@click.option('-k',
              help='number of topics to learn one LDA model using sklearn package',
              required=True)
@click.option('--n_runs',
              help='number of run to define rLDA model',
              default=100,
              show_default=True)
@click.option('--random_state_range',
              help=' list of random state, we used to run LDA models',
              default="range(n_run)",
              show_default=True)
@click.option('--n_thread',
              help='',
              default=5,
              show_default=True)
@click.option('--save_path',
              help='',
              default="",
              show_default=True)
def train_model_command(name,
                        data,
                        k,
                        n_runs=100,
                        random_state_range=None,
                        n_thread=5,
                        save_path=""):
    return train_model(name=name,
                       data=data,
                       k=k,
                       n_runs=n_runs,
                       random_state_range=random_state_range,
                       n_thread=n_thread,
                       save_path=save_path)


#### top model ####
@cli.command(name='make_topModel')
@click.option('--trains',
              help='list of train class',
              required=True)
@click.option('--data',
              help='expression data embedded in anndata format along with cells and genes/region information',
              required=True)
@click.option('--n_top_genes',
              help='Number of highly-variable genes to keep',
              default=50,
              show_default=True)
@click.option('--resolution',
              help='A parameter value controlling the coarseness of the clustering. Higher values lead to more clusters.',
              default=1,
              show_default=True)
@click.option('--file_format',
              help='indicate the format of plot',
              default="pdf",
              show_default=True)
@click.option('--save_path',
              help='directory you want to use to save pickle file',
              default="",
              show_default=True)
def make_topModel_command(trains,
                          data,
                          n_top_genes=50,
                          resolution=1,
                          file_format="pdf",
                          save_path=""):
    return make_topModel(trains=trains,
                         data=data,
                         n_top_genes=n_top_genes,
                         resolution=resolution,
                         file_format=file_format,
                         save_path=save_path)


#### top model ####
@cli.command(name='make_topModel')
@click.option('--top_model',
              help='top model',
              required=True)
@click.option('--data',
              help='expression data embedded in anndata format along with cells and genes/region information',
              required=True)
@click.option('--colors_topics',
              help='dataframe that mapped colored to topics',
              default="assign color randomly",
              show_default=True)
@click.option('--save_path',
              help='directory you want to use to save pickle file',
              default="",
              show_default=True)
def make_analysis_class_command(top_model,
                                data,
                                colors_topics=None,
                                save_path=""):
    return make_analysis_class(top_model=top_model,
                               data=data,
                               colors_topics=colors_topics,
                               save_path=save_path)
