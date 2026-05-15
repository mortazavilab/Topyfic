import argparse

import Topyfic

from common import ensure_output_dir, load_adata_inputs, save_path_arg


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-seed Topyfic Train object")
    parser.add_argument('--name', required=True)
    parser.add_argument('--adata-path', required=True)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--random-state', required=True, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--max-iter', default=5, type=int)
    parser.add_argument('--n-jobs', default=1, type=int)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    adata = load_adata_inputs([args.adata_path])

    train = Topyfic.Train(
        name=f"{args.name}_{args.k}_{args.random_state}",
        k=args.k,
        n_runs=1,
        random_state_range=[args.random_state],
    )
    train.run_LDA_models(
        adata,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        n_thread=1,
    )
    train.save_train(save_path=save_path_arg(output_dir))


if __name__ == '__main__':
    main()