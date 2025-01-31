from argparse import ArgumentParser
import json
import logging

from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def generate_splits(data_root, save_path, val_size=1000, test_size=1000, max_models=None):
    # Get the current file's path
    current_file = Path(__file__)

    # Get the parent of the parent directory
    data_path = os.path.join(str(current_file.parent.parent.parent.parent),data_root)
    save_path = os.path.join(str(current_file.parent.parent.parent.parent),save_path)
    data_root = Path(data_path)
    save_path = Path(save_path)
    data_split = defaultdict(list)
    all_files = [p.as_posix() for p in data_root.glob("**/*.pth")]
    if max_models is not None:
        all_files = all_files[:max_models]

    # test split
    train_files, test_files = train_test_split(all_files, test_size=test_size)
    data_split["test"] = test_files

    # val split
    train_files, val_files = train_test_split(train_files, test_size=val_size)
    data_split["val"] = val_files

    data_split["train"] = train_files

    logging.info(f"train size: {len(data_split['train'])}, "
                 f"val size: {len(data_split['val'])}, test size: {len(data_split['test'])}")

    with open(save_path, "w") as file:
        json.dump(data_split, file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--set_type",type=str,default='cifar')
    args = parser.parse_args()
    set_type = args.set_type
    args.data_root = os.path.join("data", "datasets/samples", f"{set_type}_models")
    args.save_path = os.path.join( "data", "datasets/samples", f"{set_type}_models_processed.json")
    generate_splits(args.data_root, args.save_path, val_size=None, test_size=None, max_models=None)
    
