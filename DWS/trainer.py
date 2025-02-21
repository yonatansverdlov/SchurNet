import json
import logging
from argparse import ArgumentParser
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append('../deepalign')
from distutils.util import strtobool
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import wandb
from sklearn.metrics import f1_score
from tqdm import trange

from deepalign.losses.mlp_losses import calc_lmc_loss, calc_recon_loss, calc_gt_perm_loss
from deepalign.utils import extract_pred
from utils.utils import (
    common_parser, count_parameters, get_device, set_logger, set_seed, str2bool,
)
from utils.data_utils import MultiViewMatchingBatch, MatchingModelsDataset
from deepalign.sinkhorn import matching
from deepalign import DWSMatching
from utils.data_utils.image_data import get_mnist_dataloaders, get_cifar10_dataloaders

set_logger()



@torch.no_grad()
def evaluate(model, loader, image_loader, add_task_loss=True, add_l2_loss=True):
    model.eval()

    perm_loss = 0.0
    recon_loss = 0.
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    recon_losses, baseline_losses, hard_recon_losses, sink_ours_losses, sink_random_losses = [], [], [], [], []
    for j, batch in enumerate(loader):
        with torch.no_grad():
          image_batch = next(iter(image_loader))
          image_batch = tuple(t.to(device) for t in image_batch)
          batch: MultiViewMatchingBatch = batch.to(device)
  
          input_0 = (batch.weights_view_0, batch.biases_view_0)
          input_1 = (batch.weights_view_1, batch.biases_view_1)
          perm_input_0 = (batch.perm_weights_view_0, batch.perm_biases_view_0)
          
          out_0, out_1 = model((input_0,input_1))
          perm_out_0, _ = model((perm_input_0,perm_input_0))
  
          pred_matrices_perm_0 = extract_pred(
              out_0,
              perm_out_0,
          )
  
          pred_matrices = extract_pred(
              out_0,
              out_1,
          )
  
          # loss from GT permutations
          curr_gt_loss = calc_gt_perm_loss(
              pred_matrices_perm_0, batch.perms_view_0, criterion=args.loss, device=device
          )
  
          # reconstruction loss
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
              device=device,
              image_flatten_size=image_flatten_size,
          )
  
          # reconstruction loss and images
          results = calc_lmc_loss(
              pred_matrices if not args.sanity else pred_matrices_perm_0,
              input_0,
              input_1 if not args.sanity else perm_input_0,
              image_batch=image_batch,
              sinkhorn_project=True,
              n_sinkhorn_iter=args.n_sink,
              device=device,
              image_flatten_size=image_flatten_size,
          )
  
          recon_losses.append(results["soft"]["losses"])
          hard_recon_losses.append(results["hard"]["losses"])
          baseline_losses.append(results["no_alignment"]["losses"])
  
          perm_loss += curr_gt_loss.item()
          recon_loss += curr_recon_loss.item()
  
          curr_correct = 0.
          curr_gts = []
          curr_preds = []
  
          for pred, gt_perm in zip(pred_matrices_perm_0, batch.perms_view_0):
              pred = matching(pred).to(device)
              curr_correct += ((pred.argmax(1)).eq(gt_perm) * 1.0).mean().item()
              curr_preds.append(pred.argmax(1).reshape(-1))
              curr_gts.append(gt_perm.reshape(-1))
  
          total += 1
          correct += (curr_correct / len(pred_matrices_perm_0))
          predicted.extend(curr_preds)
          gt.extend(curr_gts)

    predicted = torch.cat(predicted).int()
    gt = torch.cat(gt).int()

    avg_loss = perm_loss / total
    avg_acc = correct / total
    recon_loss = recon_loss / total

    f1 = f1_score(predicted.cpu().detach().numpy(), gt.cpu().detach().numpy(), average="macro")

    # LMC losses
    lmc_losses = dict(
        soft_alignment=np.stack(recon_losses).mean(0),  # NOTE: this is the soft alignment loss.
        no_alignment=np.stack(baseline_losses).mean(0),
        alignment=np.stack(hard_recon_losses).mean(0),
    )

    return dict(
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        recon_loss=recon_loss,
        predicted=predicted,
        gt=gt,
        f1=f1,
        lmc_losses=lmc_losses,
    )

def main(
    path,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
    eval_every: int,
    add_common:bool,
):
    # losses
    add_l2_loss = True if args.recon_loss in ["l2", "both"] else False
    add_task_loss = True if args.recon_loss in ["lmc", "both"] else False
    logging.info(f"Using {args.recon_loss} loss (task loss: {add_task_loss}, l2 loss: {add_l2_loss})")

    # load dataset
    train_set = MatchingModelsDataset(
        path=path,
        split="train",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )
    val_set = MatchingModelsDataset(
        path=path,
        split="val",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )
    test_set = MatchingModelsDataset(
        path=path,
        split="test",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # todo: add image args to argparse
    get_loaders = dict(mnist=get_mnist_dataloaders, cifar10=get_cifar10_dataloaders)[args.data_name]
    train_image_loader, val_image_loader, test_image_loader = get_loaders(
        args.image_data_path, batch_size=args.image_batch_size
    )

    logging.info(
        f"train size {len(train_set)}, "
        f"val size {len(val_set)}, "
        f"test size {len(test_set)}"
    )

    batch = next(iter(train_loader))
    weight_shapes, bias_shapes = batch.get_weight_shapes()

    logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

    model = DWSMatching(
            add_common = add_common,
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
        ).to(device)

    logging.info(f"number of parameters: {count_parameters(model)}")

    params = list(model.parameters())

    optimizer = {
        "adam": torch.optim.Adam(
            [
                dict(params=params, lr=lr),
            ],
            lr=lr,
            weight_decay=5e-4,
        ),
        "sgd": torch.optim.SGD(params, lr=lr, weight_decay=5e-4, momentum=0.9),
        "adamw": torch.optim.AdamW(
            params=params, lr=lr, amsgrad=True, weight_decay=args.wd
        ),
    }[args.optim]

    def save_model(sd):
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

    epoch_iter = trange(epochs)
    best_test_results, best_val_results = None, None
    test_acc, test_loss = -1.0, -1.0
    best_val_recon_loss = 1e6
    best_sd = model.state_dict()
    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch: MultiViewMatchingBatch = batch.to(device)
            image_batch = next(iter(train_image_loader))
            image_batch = tuple(t.to(device) for t in image_batch)

            input_0 = (batch.weights_view_0, batch.biases_view_0)
            input_1 = (batch.weights_view_1, batch.biases_view_1)
            perm_input_0 = (batch.perm_weights_view_0, batch.perm_biases_view_0)

            out_0, out_1 = model((input_0,input_1))
            perm_out_0, _ = model((perm_input_0, perm_input_0))

            pred_matrices_perm_0 = extract_pred(
                out_0,
                perm_out_0,
            )

            pred_matrices = extract_pred(
                out_0,
                out_1,
            )

            # loss from GT permutations
            gt_perm_loss = calc_gt_perm_loss(
                pred_matrices_perm_0, batch.perms_view_0, criterion=args.loss, device=device
            )

            # reconstruction loss
            recon_loss = calc_recon_loss(
                pred_matrices if not args.sanity else pred_matrices_perm_0,
                input_0,
                input_1 if not args.sanity else perm_input_0,
                image_batch=image_batch,
                sinkhorn_project=True,   # if we perms are already bi-stochastic we don't need to do anything
                n_sinkhorn_iter=args.n_sink,
                add_task_loss=add_task_loss,
                add_l2_loss=add_l2_loss,
                device=device,
                image_flatten_size=image_flatten_size,
            )

            loss = gt_perm_loss * args.supervised_loss_weight + recon_loss * args.recon_loss_weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.wandb:
                log = {
                    "train/loss": loss.item(),
                    "train/supervised_loss": gt_perm_loss.item(),
                    "train/recon_loss": recon_loss.item(),
                }
                wandb.log(log)

            epoch_iter.set_description(
                f"[{epoch} {i+1}], train loss: {loss.item():.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}"
            )

        if (epoch + 1) % eval_every == 0:
            val_loss_dict = evaluate(
                model, val_loader, image_loader=val_image_loader,
                add_task_loss=add_task_loss, add_l2_loss=add_l2_loss,
            )
            test_loss_dict = evaluate(
                model, test_loader, image_loader=test_image_loader,
                add_task_loss=add_task_loss, add_l2_loss=add_l2_loss,
            )
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]

            best_val_criteria = val_loss_dict["recon_loss"] <= best_val_recon_loss

            if best_val_criteria:
                best_val_recon_loss = val_loss_dict["recon_loss"]
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict
                best_sd = model.state_dict()
                if args.save_model:
                    save_model(best_sd)

            if args.wandb:
                # LMC plot
                x = torch.linspace(0.0, 1.0, len(test_loss_dict["lmc_losses"]["alignment"])).numpy().tolist()
                for k, v in test_loss_dict["lmc_losses"].items():
                    plt.plot(x, v, label=k)
                plt.legend()

                log = {
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/f1": val_loss_dict["f1"],
                    "val/recon_loss": val_loss_dict["recon_loss"],
                    "val/best_loss": best_val_results["avg_loss"],
                    "val/best_acc": best_val_results["avg_acc"],
                    "val/best_f1": best_val_results["f1"],
                    "val/best_recon_loss": best_val_results["recon_loss"],
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/f1": test_loss_dict["f1"],
                    "test/recon_loss": test_loss_dict["recon_loss"],
                    "test/best_loss": best_test_results["avg_loss"],
                    "test/best_acc": best_test_results["avg_acc"],
                    "test/best_f1": best_test_results["f1"],
                    "test/best_recon_loss": best_test_results["recon_loss"],
                    "epoch": epoch,
                    "interpolation_loss": plt,
                }
                wandb.log(log)
                plt.close()

    if args.save_model:
        save_model(best_sd)
    return best_test_results
def process_and_plot(data, save_dir="plots"):
    """
    Processes a dictionary to print three metrics, plot three curves, and save the plot.

    Args:
    - data (dict): The dictionary containing metrics and curves to process.
    - save_dir (str): Directory to save the plot. Default is "plots".
    """
    # Print metrics
    print(f"Average Loss: {data['avg_loss']:.4f}")
    print(f"Average Accuracy: {data['avg_acc']:.4f}")
    print(f"Reconstruction Loss: {data['recon_loss']:.4f}")
    
    # Extract curves
    curves = data['lmc_losses']
    print(curves)
    # Create lambda values for x-axis (evenly spaced between 0 and 1)
    num_points = len(next(iter(curves.values())))  # Number of points in the curves
    lambdas = np.linspace(0, 1, num_points)

    # Plot each curve
    plt.figure(figsize=(10, 6))
    for curve_name, curve_values in curves.items():
        plt.plot(lambdas, curve_values, label=curve_name, marker='o')

    # Add plot details
    plt.title('Loss Curves', fontsize=16)
    plt.xlabel('Lambda', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save the plot to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(gpus):
    return "cuda" if torch.cuda.is_available() else "cpu"

from pathlib import Path
from argparse import ArgumentParser
import wandb
import logging
from distutils.util import strtobool

if __name__ == "__main__":
    path_to_proj = Path(__file__).resolve()
    parent_directory = path_to_proj.parent

    # Use common_parser as in your original code
    parser = ArgumentParser("DEEP-ALIGN MLP matching trainer", parents=[common_parser])

    parser.add_argument("--data_name", type=str, default='mnist', choices=["mnist", "cifar10"], help="Dataset to use")
    parser.add_argument("--add_common", type=lambda x: bool(strtobool(x)), nargs="?", const=True, default=False, help="Enable common functionality")

    # Parse preliminary arguments to determine dataset and add_common setting
    pre_args, _ = parser.parse_known_args()

    # Assign best hyperparameters based on dataset and add_common setting
    best_hyperparams = {
        ("mnist", True): {"lr": 0.001, "wd": 5e-05, "batch_size": 8},
        ("mnist", False): {"lr": 0.001, "wd": 1e-05, "batch_size": 8},
        ("cifar10", True): {"lr": 0.0005, "wd": 0.0001, "batch_size": 7},
        ("cifar10", False): {"lr": 0.001, "wd": 0.0001, "batch_size": 7},
    }

    selected_params = best_hyperparams.get((pre_args.data_name, pre_args.add_common), {})

    # Set default parameters using common_parser
    parser.set_defaults(
        n_epochs=100,
        batch_size=selected_params.get("batch_size", 7),
        lr=selected_params.get("lr", 0.001),
        wd=selected_params.get("wd", 1e-5),
        image_batch_size=32,
        loss="ce",
        recon_loss="both",
        optim="adamw",
        num_workers=5,
        reduction="max",
        common_reduction="max",
        dim_hidden=32,
        n_hidden=4,
        output_features=128,
        n_fc_layers=1,
        set_layer="sab",
        n_heads=8,
        statistics_path=None,
        eval_every=5,
        normalize=False,
        do_rate=0.0,
        add_skip=False,
        add_layer_skip=False,
        add_bn=True,
        save_model=False,
        diagonal=True,
        hnp_setup=True,
        sanity=False,
        init_scale=1.0,
        init_off_diag_scale=1.0,
        input_dim_downsample=8,
        recon_loss_weight=1.0,
        supervised_loss_weight=1.0,
        n_sink=20
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # wandb logging
    if args.wandb:
        name = f"mlp_cls_trainer_{args.data_name}_lr_{args.lr}_bs_{args.batch_size}_seed_{args.seed}_wd_{args.wd}_add_common_{args.add_common}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    device = get_device(gpus=args.gpu)

    logging.info(f"Using {args.data_name} dataset")
    args.data_path = f'{parent_directory}/data/samples/{args.data_name}_models_processed.json'
    args.image_data_path = f'{parent_directory}/data/samples/{args.data_name}_images'
    image_flatten_size = dict(mnist=28 * 28, cifar10=32 * 32 * 3)[args.data_name]

    test = main(
        add_common=args.add_common,
        path=args.data_path,
        epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        device=device,
    )
    process_and_plot(test)
