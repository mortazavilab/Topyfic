import argparse

from Topyfic.datasets import DEFAULT_IGVF_SUBSET_URL, materialize_igvf_subset


def parse_args():
    parser = argparse.ArgumentParser(description="Build the deterministic 1000-cell IGVF subset used by the Nextflow smoke test")
    parser.add_argument(
        "--source",
        default=DEFAULT_IGVF_SUBSET_URL,
        help="Path or URL to the full IGVF AnnData file",
    )
    parser.add_argument(
        "--output",
        default="tutorials/IGVFFI3320ZCCE/IGVFFI3320ZCCE_subset_1000.h5ad",
        help="Where to write the subset h5ad file",
    )
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = materialize_igvf_subset(
        source_path_or_url=args.source,
        output_path=args.output,
        subset_size=args.subset_size,
        seed=args.seed,
        force=args.force,
    )
    print(output_path)


if __name__ == "__main__":
    main()