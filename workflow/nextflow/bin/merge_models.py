import argparse
import json
from pathlib import Path

import pandas as pd
import scanpy as sc
import Topyfic

from common import ensure_output_dir, load_adata_inputs, optional_float, optional_int, save_path_arg


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple Topyfic datasets into a shared TopModel and Analysis")
    parser.add_argument('--names-json', required=True)
    parser.add_argument('--n-topics-json', required=True)
    parser.add_argument('--count-adata-json', required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--n-top-genes', default='None')
    parser.add_argument('--resolution', default=1.0, type=float)
    parser.add_argument('--max-iter-harmony', default=10, type=int)
    parser.add_argument('--min-cell-participation', default='None')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def find_best_n_topic(n_topics, names, workdir):
    rows = []
    for name in names:
        for n_topic in n_topics:
            topmodel_path = workdir / name / str(n_topic) / 'topmodel' / f'topModel_{name}_{n_topic}.p'
            top_model = Topyfic.read_topModel(topmodel_path.as_posix())
            rows.append({'Name': name, 'k': int(n_topic), 'N': int(top_model.N)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No topModel files were found for merge selection')

    best = {}
    for name in names:
        subset = df[df['Name'] == name].copy()
        subset['diff'] = (subset['N'] - subset['k']).abs()
        best[name] = int(subset.loc[subset['diff'].idxmin(), 'k'])

    return df, best


def main():
    args = parse_args()
    names = json.loads(args.names_json)
    n_topics = [int(value) for value in json.loads(args.n_topics_json)]
    count_adata = json.loads(args.count_adata_json)
    workdir = Path(args.workdir).resolve()
    output_dir = ensure_output_dir(args.output_dir)
    ensure_output_dir(output_dir / 'figures')
    sc.settings.figdir = (output_dir / 'figures').as_posix()

    summary, best_topics = find_best_n_topic(n_topics=n_topics, names=names, workdir=workdir)
    summary.to_csv((output_dir / 'k_N.csv').as_posix(), index=False)
    pd.DataFrame.from_dict(best_topics, orient='index', columns=['best_k']).to_csv(
        (output_dir / 'best_k.csv').as_posix()
    )

    adata = load_adata_inputs([count_adata[name] for name in names])
    trains = []
    for name in names:
        best_k = best_topics[name]
        train_path = workdir / name / str(best_k) / 'train' / f'train_{name}_{best_k}.p'
        trains.append(Topyfic.read_train(train_path.as_posix()))

    top_model, clustering, adata_topmodel = Topyfic.calculate_leiden_clustering(
        trains=trains,
        data=adata,
        n_top_genes=optional_int(args.n_top_genes),
        resolution=args.resolution,
        max_iter_harmony=args.max_iter_harmony,
        min_cell_participation=optional_float(args.min_cell_participation),
    )

    top_model.save_topModel(save_path=save_path_arg(output_dir))
    adata_topmodel.write_h5ad((output_dir / 'topic_weight_umap.h5ad').as_posix())
    clustering.to_csv((output_dir / 'topic_cluster_mapping.csv').as_posix())

    analysis = Topyfic.Analysis(Top_model=top_model)
    analysis.calculate_cell_participation(data=adata)
    analysis.save_analysis(save_path=save_path_arg(output_dir))


if __name__ == '__main__':
    main()