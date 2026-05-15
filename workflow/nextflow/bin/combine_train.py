import argparse

import Topyfic

from common import ensure_output_dir, load_adata_inputs, save_path_arg


def parse_args():
    parser = argparse.ArgumentParser(description="Combine single-seed Train objects into a reproducible Train")
    parser.add_argument('--name', required=True)
    parser.add_argument('--adata-path', required=True)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--train-file', action='append', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    adata = load_adata_inputs([args.adata_path])

    single_trains = [Topyfic.read_train(train_file) for train_file in args.train_file]
    single_trains.sort(key=lambda train: train.random_state_range[0])
    random_states = [train.random_state_range[0] for train in single_trains]

    main_train = Topyfic.Train(
        name=f"{args.name}_{args.k}",
        k=args.k,
        n_runs=len(single_trains),
        random_state_range=random_states,
    )
    main_train.combine_LDA_models(data=adata, single_trains=single_trains)
    main_train.save_train(save_path=save_path_arg(output_dir))


if __name__ == '__main__':
    main()