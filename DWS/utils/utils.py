import argparse
import json
import logging
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import wandb


def unfold_matrices(weights, biases, weight_shapes, bias_shapes):
    # todo: we assume shapes are batched
    weight_shapes = tuple([[v[0].item() for v in w] for w in weight_shapes])
    biases_shapes = tuple([[v[0].item() for v in w] for w in bias_shapes])
    unfolded_weights = [None] * len(weights)
    unfolded_biases = [None] * len(biases)
    bs = len(weights[0])
    for i, (weight, bias, w_shape, b_shape) in enumerate(zip(weights, biases, weight_shapes, biases_shapes)):
        unfolded_weights[i] = weight.view(bs, *w_shape)
        unfolded_biases[i] = bias.view(bs, *b_shape)
    return unfolded_weights, unfolded_biases


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2inttupe(v):
    return list(int(i) for i in v.split(","))


common_parser = argparse.ArgumentParser(add_help=False, description="functa parser")

common_parser.add_argument(
    "--n-epochs",
    type=int,
    default=100,
    help="num epochs",
)
common_parser.add_argument("--batch-size", type=int, default=32, help="batch size")
common_parser.add_argument(
    "--seed", type=int, default=42, help="seed value for 'set_seed'"
)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
common_parser.add_argument(
    "--save-path", type=str, default="./output", help="dir path for output file"
)
# wandb
common_parser.add_argument("--wandb", dest="wandb", action="store_true")
common_parser.add_argument("--no-wandb", dest="wandb", action="store_false")
common_parser.add_argument(
    "--wandb-entity", type=str, help="wandb entity name"
)
common_parser.add_argument(
    "--wandb-project", type=str, default="deep-align", help="wandb project name"
)
common_parser.set_defaults(wandb=True)

from pathlib import Path
def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def reshape_model_output(x, y_shape, x_shape, bs):
    return x.view(bs, y_shape, x_shape, -1).permute(0, 3, 1, 2)


def get_art_dir(args):
    art_dir = Path(args.out_dir)
    art_dir.mkdir(exist_ok=True, parents=True)

    curr = 0
    existing = [
        int(x.as_posix().split("_")[-1]) for x in art_dir.iterdir() if x.is_dir()
    ]
    if len(existing) > 0:
        curr = max(existing) + 1

    out_dir = art_dir / f"version_{curr}"
    out_dir.mkdir()

    return out_dir


def save_experiment(args, results, return_out_dir=False, save_results=True):
    out_dir = get_art_dir(args)

    json.dump(vars(args), open(out_dir / "meta.experiment", "w"))

    # loss curve
    if save_results:
        json.dump(results, open(out_dir / "results.experiment", "w"))

    if return_out_dir:
        return out_dir


def model_save(model, file=None, log_to_wandb=False):
    if file is None:
        file = BytesIO()
    torch.save({"model_state_dict": model.state_dict()}, file)
    if log_to_wandb:
        wandb.save(file.as_posix())

    return file


def model_load(model, file):
    if isinstance(file, BytesIO):
        file.seek(0)

    model.load_state_dict(
        torch.load(file, map_location=lambda storage, location: storage,use_weight = True)[
            "model_state_dict"
        ]
    )

    return model


def save_data(tensor, file, log_to_wandb=False):
    torch.save(tensor, file)
    if log_to_wandb:
        wandb.save(file.as_posix())


def load_data(file):
    return torch.load(file, map_location=lambda storage, location: storage,use_weight = True)


def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (-1, 1),
) -> torch.Tensor:
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def concat_features(a, b):
    """Takes a, b tensor and generates a new tensor of shape (bs, n, m, d_1 + d_2) where d_1 and d_2 are the feature
    dimensions of a and b respectively

    """
    bs, n, d_1 = a.shape
    bs, m, d_2 = b.shape
    # Expand dimensions for broadcasting
    expanded_a = a.unsqueeze(1)  # Shape: (bs, 1, n, d_1)
    expanded_b = b.unsqueeze(2)  # Shape: (bs, m, 1, d_2)

    # Tile to create repeated tensors
    tiled_a = expanded_a.repeat(1, m, 1, 1)  # Shape: (bs, m, n, d_1)
    tiled_b = expanded_b.repeat(1, 1, n, 1)  # Shape: (bs, m, n, d_2)

    # dot product
    # here we assume d_1 == d_2
    dot_product = tiled_a * tiled_b  # Shape: (bs, n, m, d_1)

    # Concatenate along the last dimension
    # concatenated_tensor = torch.cat((tiled_a, tiled_b, dot_product), dim=-1).permute(0, 2, 1, 3)  # Shape: (bs, n, m, d_1 + d_2 + d_1)
    concatenated_tensor = torch.cat((tiled_a, tiled_b), dim=-1).permute(0, 2, 1, 3)  # Shape: (bs, n, m, d_1 + d_2)
    # assert concatenated_tensor.shape == (bs, n, m, d_1 + d_2 + d_1)  # todo: remove this after testing
    assert concatenated_tensor.shape == (bs, n, m, d_1 + d_2)  # todo: remove this after testing
    return concatenated_tensor


def pointwise_product(a, b):
    """Takes a, b tensor and generates a new tensor of shape (bs, n, m, d_1 + d_2) where d_1 and d_2 are the feature
    dimensions of a and b respectively

    """
    bs, n, d_1 = a.shape
    bs, m, d_2 = b.shape
    # Expand dimensions for broadcasting
    expanded_a = a.unsqueeze(1)  # Shape: (bs, 1, n, d_1)
    expanded_b = b.unsqueeze(2)  # Shape: (bs, m, 1, d_2)

    # Tile to create repeated tensors
    tiled_a = expanded_a.repeat(1, m, 1, 1)  # Shape: (bs, m, n, d_1)
    tiled_b = expanded_b.repeat(1, 1, n, 1)  # Shape: (bs, m, n, d_2)

    # dot product
    # here we assume d_1 == d_2
    # dot_product = tiled_a * tiled_b  # Shape: (bs, n, m, d_1)
    return tiled_a * tiled_b


if __name__ == '__main__':
    a = torch.randn(2, 3, 4)
    b = torch.rand(2, 3, 4)
    f = concat_features(a, b)
    print()
