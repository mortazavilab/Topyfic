import argparse

import scanpy as sc
import Topyfic

from common import ensure_output_dir, load_adata_inputs, optional_float, optional_int, save_path_arg


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Topyfic TopModel from a combined Train object")
    parser.add_argument('--name', required=True)
    parser.add_argument('--adata-path', required=True)
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--n-top-genes', default='None')
    parser.add_argument('--resolution', default=1.0, type=float)
    parser.add_argument('--max-iter-harmony', default=10, type=int)
    parser.add_argument('--min-cell-participation', default='None')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    ensure_output_dir(output_dir / 'figures')
    sc.settings.figdir = (output_dir / 'figures').as_posix()

    train = Topyfic.read_train(args.train_file)
    adata = load_adata_inputs([args.adata_path])

    top_model, clustering, adata_topmodel = Topyfic.calculate_leiden_clustering(
        trains=[train],
        data=adata,
        n_top_genes=optional_int(args.n_top_genes),
        resolution=args.resolution,
        max_iter_harmony=args.max_iter_harmony,
        min_cell_participation=optional_float(args.min_cell_participation),
    )

    top_model.save_topModel(save_path=save_path_arg(output_dir))
    adata_topmodel.write_h5ad((output_dir / 'topic_weight_umap.h5ad').as_posix())
    clustering.to_csv((output_dir / 'topic_cluster_mapping.csv').as_posix())


if __name__ == '__main__':
    main()