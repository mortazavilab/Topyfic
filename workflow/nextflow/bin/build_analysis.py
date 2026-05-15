import argparse

import Topyfic

from common import ensure_output_dir, load_adata_inputs, save_path_arg


def parse_args():
    parser = argparse.ArgumentParser(description="Build an Analysis object from a TopModel")
    parser.add_argument('--adata-path', required=True)
    parser.add_argument('--topmodel-file', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    adata = load_adata_inputs([args.adata_path])
    top_model = Topyfic.read_topModel(args.topmodel_file)

    analysis = Topyfic.Analysis(Top_model=top_model)
    analysis.calculate_cell_participation(data=adata)
    analysis.save_analysis(save_path=save_path_arg(output_dir))


if __name__ == '__main__':
    main()