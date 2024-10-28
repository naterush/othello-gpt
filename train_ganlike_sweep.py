import random
import numpy as np
import torch
import torch.nn as nn

import einops
from fancy_einsum import einsum
import torch.nn.functional as F
import wandb

from transformer_lens_adapters import get_empty_transformer_lens

from othello_utils import (
    state_stack_to_one_hot_threeway,
    build_state_stack,
)

from data import get_board_seqs_int_and_str, get_valid_moves

random.seed(42)

class LinearProbe(nn.Module):
    def __init__(
            self, 
            d_model, 
            modes, 
            rows, 
            cols, 
            options
        ):
        super().__init__()
        self.probe = nn.Parameter(
            torch.randn(modes, d_model, rows, cols, options) / np.sqrt(d_model)
        )

    def forward(self, x, state_stack_one_hot):
        probe_out = einsum(
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
            x,
            self.probe,
        )
        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = einops.reduce(
            probe_log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean",
        ) * 3  # 3 is the number of options (empty, white, black)
        loss = -probe_correct_log_probs[0, :].mean(0).sum()
        return probe_out, loss
    
class WeightedLoss:
    def __init__(self, alpha=50, beta=5):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, test_loss, probe_loss):
        return self.alpha * test_loss + self.beta * probe_loss

def validate_probe(othello_gpt, linear_probe, valid_games_int, state_stack_validation, config):
    """
    Validate probe performance on validation set.
    
    Returns:
        dict: Contains validation metrics including accuracy and loss
    """
    device = next(othello_gpt.parameters()).device
    othello_gpt.eval()
    linear_probe.eval()
    
    batch_size = config["batch_size"]
    total_correct = 0
    total_samples = 0
    total_loss = 0
    pos_start = 0
    pos_end = othello_gpt.cfg.n_ctx - 0


    one_hot_function = state_stack_to_one_hot_threeway
    layer = 4
    
    
    # We validate with a random 128 games
    with torch.no_grad():
            batch_indices = torch.randint(0, len(valid_games_int), (batch_size,))
            batch_games_int = valid_games_int[batch_indices].to(device)
            state_stack = state_stack_validation[batch_indices].to(device)[:, pos_start:pos_end, :, :]
            state_stack_one_hot = one_hot_function(state_stack)
            
            # Get model activations
            _, cache = othello_gpt.run_with_cache(
                batch_games_int[:, :-1], return_type=None
            )
            resid_post = cache["resid_post", layer]
            
            # Get probe predictions and loss
            probe_out, probe_loss = linear_probe(resid_post, state_stack_one_hot)
            
            # Calculate accuracy
            pred = probe_out.argmax(dim=-1)  # Shape: [modes, batch, pos, rows, cols]
            true = state_stack_one_hot.argmax(dim=-1)  # Shape: [batch, pos, rows, cols]
            
            correct = (pred[0] == true).sum().item()  # Only look at first mode
            total_correct += correct
            total_samples += true.numel()
            total_loss += probe_loss.item() * batch_games_int.size(0)
    
    metrics = {
        "accuracy": total_correct / total_samples,
        "loss": total_loss / len(valid_games_int)
    }
    
    return metrics

def get_accuracy_of_othello_gpt(othello_gpt, board_seqs_int, board_seqs_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_correct = 0
    total_predictions = 0

    othello_gpt.eval()  # Set the model to evaluation mode
    batch_size = 32

    with torch.no_grad():
        batch_indices = torch.randint(0, len(board_seqs_int), (batch_size,))
        batch_seqs_int = board_seqs_int[batch_indices].to(device)
        batch_seqs_string = board_seqs_string[batch_indices]
        valid_moves = get_valid_moves(batch_seqs_string).to(device) # Batch x 60 x 61

        # Get model predictions
        logits = othello_gpt(batch_seqs_int[:, :-1])
        predictions = logits.argmax(dim=-1) # Batch x 59

        # For every game in the batch, for every move in the game, check if it's valid
        for j in range(predictions.shape[0]):  # Iterate over the games
            for k in range(predictions.shape[1]):
                if valid_moves[j, k + 1, predictions[j, k]] == 1:
                    total_correct += 1
                total_predictions += 1

    accuracy = total_correct / total_predictions
    return accuracy

def train_with_config():
    # This function will be called by wandb.agent with the sweep config
    def train(config=None):
        with wandb.init(config=config) as run:
            # Access the config through wandb.config
            config = wandb.config
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Model setup
            othello_gpt = get_empty_transformer_lens().to(device)
            linear_probe = LinearProbe(othello_gpt.cfg.d_model, 1, 8, 8, 3).to(device)
            
            # Optimizers
            optimizer_class = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.SGD
                
            probe_optimizer = optimizer_class(
                linear_probe.parameters(), lr=config.probe_lr, weight_decay=config.wd
            )
            othello_gpt_optimizer = optimizer_class(
                othello_gpt.parameters(), lr=config.model_lr, weight_decay=config.wd
            )

            pos_start = 0
            pos_end = othello_gpt.cfg.n_ctx - 0

            # Initialize loss function with config parameters
            loss_fn = WeightedLoss(alpha=config.alpha, beta=config.beta)

            # Get data
            board_seqs_int, board_seqs_string, board_seqs_int_validation, board_seqs_string_validation, state_stack_train, state_stack_validation = get_board_seqs_int_and_str(
                config.num_games if hasattr(config, "num_games") else 1000
            )
            one_hot_function = state_stack_to_one_hot_threeway

            layer = 4
            
            # Training setup
            train_size = board_seqs_int.shape[0]
            batch_size = config.batch_size

            global_step = 0
            
            for epoch in range(config.epochs):
                # Train probe
                for _ in range(config.probe_steps_per_epoch):
                    # Clear gradients at the start of each epoch - since they might roll over
                    probe_optimizer.zero_grad()
                    othello_gpt_optimizer.zero_grad()

                    batch_indices = torch.randint(0, train_size, (batch_size,))
                    games_int_batch = board_seqs_int[batch_indices].to(device)
                    state_stack = state_stack_train[batch_indices].to(device)[:, pos_start:pos_end, :, :]
                    state_stack_one_hot = one_hot_function(state_stack)

                    with torch.inference_mode():
                        _, cache = othello_gpt.run_with_cache(
                            games_int_batch[:, :-1], return_type=None
                        )
                        resid_post = cache["resid_post", layer]

                    linear_probe.train()
                    resid_post = resid_post.clone().detach().requires_grad_(True)
                    _, probe_loss = linear_probe(resid_post, state_stack_one_hot)

                    probe_loss.backward()
                    probe_optimizer.step()  

                    wandb.log({
                        "probe/train_loss": probe_loss.item(),
                        "epoch": epoch,
                        "global_step": global_step
                    })
                    global_step += 1

                # Train model
                for _ in range(config.model_steps_per_epoch):
                    # Clear gradients at the start of each epoch - since they might roll over
                    probe_optimizer.zero_grad()
                    othello_gpt_optimizer.zero_grad()

                    batch_indices = torch.randint(0, train_size, (batch_size,))
                    games_int_batch = board_seqs_int[batch_indices].to(device)
                    state_stack = state_stack_train[batch_indices].to(device)[:, pos_start:pos_end, :, :]
                    state_stack_one_hot = one_hot_function(state_stack)

                    othello_gpt.train()
                    logits, cache = othello_gpt.run_with_cache(
                        games_int_batch[:, :-1], return_type="logits"
                    )
                    resid_post = cache["resid_post", layer]

                    logits_reshaped = logits.view(-1, logits.size(-1))
                    targets_reshaped = games_int_batch[:, 1:].reshape(-1)
                    test_loss = F.cross_entropy(logits_reshaped, targets_reshaped, ignore_index=0)
                    
                    _, probe_loss = linear_probe(resid_post, state_stack_one_hot)
                    total_loss = loss_fn(test_loss, probe_loss)

                    total_loss.backward()
                    othello_gpt_optimizer.step()

                    wandb.log({
                        "model/test_loss_raw": test_loss.item(),
                        "model/probe_loss_raw": probe_loss.item(),
                        "model/total_loss": total_loss.item(),
                        "model/test_loss": test_loss.item() * config.alpha,
                        "model/probe_loss": probe_loss.item() * config.beta,
                        "epoch": epoch,
                        "global_step": global_step
                    })
                    global_step += 1

                # Validation step
                if epoch % config.valid_every == 0:
                    othello_accuracy = get_accuracy_of_othello_gpt(
                        othello_gpt, board_seqs_int, board_seqs_string
                    )
                    
                    probe_val_metrics = validate_probe(
                        othello_gpt, 
                        linear_probe,
                        board_seqs_int_validation,
                        state_stack_validation,
                        {"batch_size": config.batch_size}
                    )

                    wandb.log({
                        "val/othello_accuracy": othello_accuracy,
                        "val/probe_accuracy": probe_val_metrics["accuracy"],
                        "val/probe_loss": probe_val_metrics["loss"],
                        "epoch": epoch,
                        "global_step": global_step
                    })
                    global_step += 1

            return othello_gpt, linear_probe

    return train

# Example sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val/othello_accuracy', 'goal': 'maximize'},
    'parameters': {
        'probe_lr': {'values': [1e-4, 1e-3]},
        'model_lr': {'values': [1e-4, 1e-3]},
        'probe_steps_per_epoch': {'values': [10, 20]},
        'model_steps_per_epoch': {'values': [10, 20]},
        'alpha': {'values': [1, 10, 50]},
        'beta': {'values': [-1, -10, -50]},
        'batch_size': {'value': 512},
        'wd': {'value': 0.01},
        'epochs': {'value': 10},
        'valid_every': {'value': 1},
        'num_games': {'value': int(1e5)},
        'optimizer': {'values': ['sdg', 'adamw']},
    }
}

if __name__ == "__main__":
    # Start sweep
    sweep_id = wandb.sweep(sweep_config, project="othello-privacy")
    wandb.agent(sweep_id, train_with_config())

    """
    # Train with a single configuration
    config = {
        "probe_lr": 1e-4,
        "model_lr": 1e-4,
        "probe_steps_per_epoch": 5,
        "model_steps_per_epoch": 5,
        "alpha": 50,
        "beta": -50,
        "batch_size": 512,
        "wd": 0.01,
        "epochs": 5,
        "valid_every": 1,
        "num_games": int(1e5),
        "optimizer": "sdg",
    }
    import cProfile
    def run_training():
        train_with_config()(config)

    # Profile the run
    cProfile.run('run_training()', sort='cumtime')
    """

