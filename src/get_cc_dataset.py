import os, argparse
from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--download_dir",
        type=str,
        required=True,
        help="Path to store the downloaded CommonCanvas dataset."
    )
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)

    # download the dataset using huggingface datasets library
    ds = load_dataset(
        "common-canvas/commoncatalog-cc-by-nd",
        cache_dir=args.download_dir
    )