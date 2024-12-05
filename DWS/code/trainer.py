import json
import logging
from argparse import ArgumentParser
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score
from tqdm import trange

from deepalign.losses.mlp_losses import calc_lmc_loss, calc_recon_loss, calc_gt_perm_loss
from deepalign.utils import extract_pred
from utils import (
    common_parser, count_parameters, get_device, set_logger, set_seed, str2bool,
)
from utils.data_utils import MultiViewMatchingBatch, MatchingModelsDataset
from deepalign.sinkhorn import matching
from deepalign import DWSMatching
from utils.data_utils.image_data import get_mnist_dataloaders, get_cifar10_dataloaders
import logging
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import logging
import os
from argparse import ArgumentParser

set_logger()

import torch
import numpy as np
from sklearn.metrics import f1_score

@torch.no_grad()
def evaluate(args, model, loader, image_loader, add_task_loss=True, add_l2_loss=True):
    """
    Evaluate the model on a given dataset and image loader.

    Args:
        model: The model to evaluate.
        loader: DataLoader for the dataset to evaluate on.
        image_loader: DataLoader for images.
        add_task_loss (bool): Whether to include task-specific loss in reconstruction loss.
        add_l2_loss (bool): Whether to include L2 loss in reconstruction loss.

    Returns:
        dict: Evaluation metrics including average loss, accuracy, reconstruction loss, F1 score, and LMC losses.
    """
    model.eval()

    perm_loss, recon_loss, correct, total = 0.0, 0.0, 0.0, 0.0
    predicted, gt = [], []
    recon_losses, baseline_losses, hard_recon_losses = [], [], []
    # Iterate over batches
    for batch in loader:
        # Fetch corresponding image batch
        image_batch = next(iter(image_loader))
        image_batch = tuple(t.to(args.device) for t in image_batch)
        batch: MultiViewMatchingBatch = batch.to(args.device)

        # Prepare inputs for model
        input_0 = (batch.weights_view_0, batch.biases_view_0)
        input_1 = (batch.weights_view_1, batch.biases_view_1)
        perm_input_0 = (batch.perm_weights_view_0, batch.perm_biases_view_0)

        # Forward pass
        out_0, out_1 = model((input_0, input_1))
        perm_out_0, _ = model((perm_input_0, perm_input_0))

        # Extract predictions
        pred_matrices_perm_0 = extract_pred(out_0, perm_out_0)
        pred_matrices = extract_pred(out_0, out_1)

        # Calculate GT permutation loss
        curr_gt_loss = calc_gt_perm_loss(
            pred_matrices_perm_0, batch.perms_view_0, criterion=args.loss, device=args.device
        )

        # Calculate reconstruction loss
        curr_recon_loss = calc_recon_loss(
            pred_matrices if not args.sanity else pred_matrices_perm_0,
            input_0,
            input_1 if not args.sanity else perm_input_0,
            image_batch=image_batch,
            sinkhorn_project=True,
            n_sinkhorn_iter=args.n_sink,
            add_task_loss=add_task_loss,
            add_l2_loss=add_l2_loss,
            alpha=0.5,
            eval_mode=True,
            device=args.device,
            image_flatten_size=args.image_flatten_size,
        )

        # Calculate LMC loss
        results = calc_lmc_loss(
            pred_matrices if not args.sanity else pred_matrices_perm_0,
            input_0,
            input_1 if not args.sanity else perm_input_0,
            image_batch=image_batch,
            sinkhorn_project=True,
            n_sinkhorn_iter=args.n_sink,
            device=args.device,
            image_flatten_size=args.image_flatten_size,
        )

        # Store losses
        recon_losses.append(results["soft"]["losses"])
        hard_recon_losses.append(results["hard"]["losses"])
        baseline_losses.append(results["no_alignment"]["losses"])

        # Accumulate losses
        perm_loss += curr_gt_loss.item()
        recon_loss += curr_recon_loss.item()

        # Calculate accuracy
        curr_correct = 0.0
        curr_preds, curr_gts = [], []
        for pred, gt_perm in zip(pred_matrices_perm_0, batch.perms_view_0):
            pred = matching(pred).to(args.device)
            curr_correct += ((pred.argmax(1)).eq(gt_perm) * 1.0).mean().item()
            curr_preds.append(pred.argmax(1).reshape(-1))
            curr_gts.append(gt_perm.reshape(-1))

        total += 1
        correct += (curr_correct / len(pred_matrices_perm_0))
        predicted.extend(curr_preds)
        gt.extend(curr_gts)

    # Calculate metrics
    predicted = torch.cat(predicted).int()
    gt = torch.cat(gt).int()

    avg_loss = perm_loss / total
    avg_acc = correct / total
    recon_loss = recon_loss / total
    f1 = f1_score(predicted.cpu().detach().numpy(), gt.cpu().detach().numpy(), average="macro")

    # Calculate LMC losses
    lmc_losses = {
        "soft_alignment": np.stack(recon_losses).mean(0),
        "no_alignment": np.stack(baseline_losses).mean(0),
        "alignment": np.stack(hard_recon_losses).mean(0),
    }

    # Return evaluation results
    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "recon_loss": recon_loss,
        "predicted": predicted,
        "gt": gt,
        "f1": f1,
        "lmc_losses": lmc_losses,
    }

def save_model(args, sd, weight_shapes, bias_shapes):
    path = Path(args.save_path)
    artifact_path = path / args.recon_loss / f"{args.seed}"
    artifact_path.mkdir(parents=True, exist_ok=True)
    # save model
    torch.save(sd, artifact_path / f"model.pth")

    with open(artifact_path / "args.json", "w") as f:
        json.dump(vars(args), f)

    model_args = dict(
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        input_features=1,
        hidden_dim=args.dim_hidden,
        n_hidden=args.n_hidden,
        reduction=args.reduction,
        n_fc_layers=args.n_fc_layers,
        set_layer=args.set_layer,
        output_features=args.output_features
        if args.output_features is not None
        else args.dim_hidden,
        input_dim_downsample=args.input_dim_downsample,
        add_skip=args.add_skip,
        add_layer_skip=args.add_layer_skip,
        init_scale=args.init_scale,
        init_off_diag_scale_penalty=args.init_off_diag_scale,
        bn=args.add_bn,
        diagonal=args.diagonal,
        hnp_setup=args.hnp_setup,
    )
    with open(artifact_path / "model_config.json", "w") as f:
        json.dump(model_args, f)

def main(args: dict):
    """
    Main training and evaluation loop for the model.

    Args:
        args (dict): Dictionary of arguments and configurations.
        device (str): Device to use for training ('cpu' or 'cuda').

    """
    # Determine loss components
    device = args.device
    add_l2_loss = args.recon_loss in ["l2", "both"]
    add_task_loss = args.recon_loss in ["lmc", "both"]
    logging.info(f"Using {args.recon_loss} loss (task loss: {add_task_loss}, l2 loss: {add_l2_loss})")

    # Load datasets
    train_set, val_set, test_set = load_datasets(args)
    train_loader, val_loader, test_loader = create_data_loaders(train_set, val_set, test_set, args)

    # Load image loaders
    get_loaders = {"mnist": get_mnist_dataloaders, "cifar": get_cifar10_dataloaders}[args.set_type]
    train_image_loader, val_image_loader, test_image_loader = get_loaders(
        args.imgs_path, batch_size=args.image_batch_size
    )

    logging.info(f"train size: {len(train_set)}, val size: {len(val_set)}, test size: {len(test_set)}")

    # Get weight and bias shapes from the first batch
    batch = next(iter(train_loader))
    weight_shapes, bias_shapes = batch.get_weight_shapes()
    logging.info(f"Weight shapes: {weight_shapes}, Bias shapes: {bias_shapes}")

    # Initialize model
    model = initialize_model(weight_shapes, bias_shapes, args).to(device)
    logging.info(f"Number of parameters: {count_parameters(model)}")

    # Define optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        amsgrad=True,
        weight_decay=args.wd,
    )

    # Training loop
    best_results = {"val": None, "test": None, "model_state": None}
    best_val_recon_loss = float("inf")
    epoch_iter = trange(args.n_epochs)

    for epoch in epoch_iter:
        train_epoch(train_loader, train_image_loader, model, optimizer, device, args, epoch_iter, epoch)

        # Evaluate on validation and test datasets
        if (epoch + 1) % args.eval_every == 0:
            val_results = evaluate_model(loader = val_loader, image_loader = val_image_loader, model = model, args = args)
            test_results = evaluate_model(loader = test_loader, image_loader = test_image_loader, model = model, args = args)

            # Update best results based on validation reconstruction loss
            if val_results["recon_loss"] <= best_val_recon_loss:
                best_val_recon_loss = val_results["recon_loss"]
                best_results.update({"val": val_results, "test": test_results, "model_state": model.state_dict()})

                if args.save_model:
                    save_model(best_results["model_state"])

            # Log metrics
            log_metrics(val_results, test_results, best_results, args, epoch)

    if args.save_model:
        save_model(best_results["model_state"])
    return test_results


def load_datasets(args):
    """Load train, validation, and test datasets."""
    print("hi")
    return (
        MatchingModelsDataset(args.data_path, split="train", normalize=args.normalize, statistics_path=args.statistics_path),
        MatchingModelsDataset(args.data_path, split="val", normalize=args.normalize, statistics_path=args.statistics_path),
        MatchingModelsDataset(args.data_path, split="test", normalize=args.normalize, statistics_path=args.statistics_path),
    )


def create_data_loaders(train_set, val_set, test_set, args):
    """Create data loaders for train, validation, and test sets."""
    return (
        torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True),
        torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
        torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    )


def initialize_model(weight_shapes, bias_shapes, args):
    """Initialize and return the model."""
    return DWSMatching(
        add_common=args.shared,
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        input_features=1,
        hidden_dim=args.dim_hidden,
        n_hidden=args.n_hidden,
        reduction=args.reduction,
        n_fc_layers=args.n_fc_layers,
        set_layer=args.set_layer,
        output_features=args.output_features or args.dim_hidden,
        input_dim_downsample=args.input_dim_downsample,
        add_skip=args.add_skip,
        add_layer_skip=args.add_layer_skip,
        init_scale=args.init_scale,
        init_off_diag_scale_penalty=args.init_off_diag_scale,
        bn=args.add_bn,
        diagonal=args.diagonal,
        hnp_setup=args.hnp_setup,
    )


def train_epoch(train_loader, train_image_loader, model, optimizer, device, args, epoch_iter, epoch):
    """Train the model for one epoch."""
    for i, batch in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        batch: MultiViewMatchingBatch = batch.to(device)
        image_batch = next(iter(train_image_loader))
        image_batch = tuple(t.to(device) for t in image_batch)

        input_0 = (batch.weights_view_0, batch.biases_view_0)
        input_1 = (batch.weights_view_1, batch.biases_view_1)
        perm_input_0 = (batch.perm_weights_view_0, batch.perm_biases_view_0)

        # Forward pass
        out_0, out_1 = model((input_0, input_1))
        perm_out_0, _ = model((perm_input_0, perm_input_0))

        pred_matrices_perm_0 = extract_pred(out_0, perm_out_0)
        pred_matrices = extract_pred(out_0, out_1)

        # Compute losses
        gt_perm_loss = calc_gt_perm_loss(pred_matrices_perm_0, batch.perms_view_0, criterion=args.loss, device=device)
        recon_loss = calc_recon_loss(
            pred_matrices if not args.sanity else pred_matrices_perm_0,
            input_0,
            input_1 if not args.sanity else perm_input_0,
            image_batch=image_batch,
            sinkhorn_project=True,
            n_sinkhorn_iter=args.n_sink,
            add_task_loss=args.recon_loss in ["lmc", "both"],
            add_l2_loss=args.recon_loss in ["l2", "both"],
            device=device,
            image_flatten_size=args.image_flatten_size,
        )

        # Backpropagation
        loss = gt_perm_loss * args.supervised_loss_weight + recon_loss * args.recon_loss_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if args.wandb:
            wandb.log({"train/loss": loss.item(), "train/supervised_loss": gt_perm_loss.item(), "train/recon_loss": recon_loss.item()})

        epoch_iter.set_description(f"[{epoch} {i + 1}], train loss: {loss.item():.3f}")


def evaluate_model(loader, image_loader, model,  args):
    """Evaluate the model on a given dataset and return results."""
    return evaluate(
        model = model,
        loader = loader,
        image_loader=image_loader,
        add_task_loss=args.recon_loss in ["lmc", "both"],
        add_l2_loss=args.recon_loss in ["l2", "both"],
        args=args
    )


def log_metrics(val_results, test_results, best_results, args, epoch):
    """Log metrics to console or wandb."""
    if args.wandb:
        x = torch.linspace(0.0, 1.0, len(test_results["lmc_losses"]["alignment"])).numpy().tolist()
        for k, v in test_results["lmc_losses"].items():
            plt.plot(x, v, label=k)
        plt.legend()

        wandb.log({
            "val/loss": val_results["avg_loss"],
            "val/acc": val_results["avg_acc"],
            "val/f1": val_results["f1"],
            "val/recon_loss": val_results["recon_loss"],
            "test/loss": test_results["avg_loss"],
            "test/acc": test_results["avg_acc"],
            "test/f1": test_results["f1"],
            "test/recon_loss": test_results["recon_loss"],
            "epoch": epoch,
        })
        plt.close()

def configure_parser():
    """
    Configure and return the argument parser with all necessary arguments.
    """
    parser = ArgumentParser("DEEP-ALIGN MLP matching trainer", parents=[common_parser])
    parser.set_defaults(
        lr=0.001,
        n_epochs=100,
        batch_size=8,
        wd=1e-5,
        add_common=True,
    )
    parser.add_argument("--set_type", type=str, default="cifar", choices=["mnist", "cifar"], help="Dataset type")
    parser.add_argument("--image-batch-size", type=int, default=32, help="Image batch size")
    parser.add_argument("--loss", type=str, choices=["ce", "mse"], default="ce", help="Loss function for permutations")
    parser.add_argument("--recon-loss", type=str, choices=["l2", "lmc", "both"], default="both", help="Reconstruction loss type")
    parser.add_argument("--optim", type=str, choices=["adam", "sgd", "adamw"], default="adamw", help="Optimizer type")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--reduction", type=str, choices=["mean", "sum", "max", "attn"], default="max", help="Reduction strategy")
    parser.add_argument("--common_reduction", type=str, choices=["mean", "sum", "max", "attn"], default="max", help="Common reduction strategy")
    parser.add_argument("--dim-hidden", type=int, default=32, help="Dimension of hidden layers")
    parser.add_argument("--n-hidden", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--output-features", type=int, default=128, help="Number of output features")
    parser.add_argument("--n-fc-layers", type=int, default=1, help="Number of fully connected layers in each block")
    parser.add_argument("--set-layer", type=str, choices=["sab", "ds"], default="sab", help="Set layer type")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--statistics-path", type=str, default=None, help="Path to dataset statistics")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluation frequency")
    parser.add_argument("--normalize", type=str2bool, default=False, help="Normalize data")
    parser.add_argument("--do-rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--add-skip", type=str2bool, default=False, help="Add skip connection")
    parser.add_argument("--add-layer-skip", type=str2bool, default=False, help="Add layer-wise skip connections")
    parser.add_argument("--add-bn", type=str2bool, default=True, help="Add batch normalization")
    parser.add_argument("--save-model", type=str2bool, default=False, help="Save model artifacts")
    parser.add_argument("--diagonal", type=str2bool, default=True, help="Use diagonal DWSNet")
    parser.add_argument("--hnp-setup", type=str2bool, default=True, help="HNP vs NP setup")
    parser.add_argument("--sanity", type=str2bool, default=False, help="Sanity check using a network and its permutation")
    parser.add_argument("--init-scale", type=float, default=1.0, help="Initialization scale")
    parser.add_argument("--init-off-diag-scale", type=float, default=1.0, help="Initialization off-diagonal scale")
    parser.add_argument("--input-dim-downsample", type=int, default=8, help="Input dimension for downsampling")
    parser.add_argument("--recon-loss-weight", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--supervised-loss-weight", type=float, default=1.0, help="Weight for supervised loss")
    parser.add_argument("--n-sink", type=int, default=20, help="Number of Sinkhorn iterations")
    parser.add_argument("--shared", type=str2bool, default=True, help="Use shared configuration for settings")
    return parser

def set_hyperparameters(args:dict):
    """
    Set hyperparameters based on dataset and shared settings.
    """
    args.lr = 0.0005 if args.set_type == "mnist" else (0.0005 if args.shared else 0.001)
    args.wd = 1e-4 if args.shared and args.set_type == "mnist" else 1e-5
    args.dim_hidden = 64 if args.set_type == "mnist" else 32


def configure_wandb(args):
    """
    Configure Weights & Biases logging if enabled.
    """
    if args.wandb:
        name = (
            f"mlp_cls_trainer_{args.data_name}_lr_{args.lr}_bs_{args.batch_size}_seed_{args.seed}_wd_{args.wd}_add_common_{args.add_common}_common_reduction_{args.common_reduction}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

def main_entry():
    """
    Main entry point for the script.
    """
    parser = configure_parser()
    args = parser.parse_args()

    set_hyperparameters(args)
    assert args.set_type in ["mnist", "cifar"], "Invalid dataset type"

    # Set seed for reproducibility
    set_seed(args.seed)

    # Configure WandB
    configure_wandb(args)

    # Select device
    device = get_device(gpus=args.gpu)
    logging.info(f"Using {args.set_type} dataset on device: {device}")
    args.device = device
    # Configure data paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data", "datasets", "samples")
    args.data_path = os.path.join(data_dir, f"{args.set_type}_models_processed.json")
    args.imgs_path = os.path.join(data_dir, f"{args.set_type}_imgs")

    # Image flatten size based on dataset
    args.image_flatten_size = {"mnist": 28 * 28, "cifar": 32 * 32 * 3}[args.set_type]

    # Execute main training and evaluation
    return main(args)

if __name__ == "__main__":
    test_results = main_entry()
    avg_loss = test_results['avg_loss']
    avg_acc = test_results['avg_acc']
    recon_loss = test_results['recon_loss']
    print(f"After training we have avg loss {avg_loss}, avg acc {avg_acc}, recon_loss {recon_loss}")
