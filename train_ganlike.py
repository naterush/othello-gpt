import json
import random
import os
import time
import numpy as np
import torch
import torch.nn as nn

import einops
from fancy_einsum import einsum
import torch.nn.functional as F
from tqdm import tqdm

from transformer_lens_adapters import get_model_transformer_lens, get_empty_transformer_lens
from transformer_lens import HookedTransformer

from othello_utils import (
    plot_single_board,
    state_stack_to_one_hot_threeway_black_white,
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
    
class NonLinearProbe(nn.Module):
    def __init__(
        self,
        d_model,
        modes,
        rows,
        cols,
        options,
        hidden_dim=64  # Added hidden dimension parameter
    ):
        super().__init__()
        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Parameter(
            torch.randn(modes, hidden_dim, rows, cols, options) / np.sqrt(hidden_dim)
        )

    def forward(self, x, state_stack_one_hot):
        # Apply hidden layer and ReLU activation
        hidden = self.relu(self.hidden_layer(x))
        
        # Use the hidden representation for the probe
        probe_out = einsum(
            "batch pos hidden_dim, modes hidden_dim rows cols options -> modes batch pos rows cols options",
            hidden,
            self.output_layer,
        )
        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = einops.reduce(
            probe_log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean",
        ) * 3  # 3 is the number of options (empty, white, black)
        loss = -probe_correct_log_probs[0, :].mean(0).sum()
        return probe_out, loss
    
    
PROBE_TYPE_TO_CLASS = {
    'linear': LinearProbe,
    'nonlinear': NonLinearProbe
}

ONE_HOT_TYPE_TO_FUNCTION = {
    "black_white": state_stack_to_one_hot_threeway_black_white,
    "mine_theirs": state_stack_to_one_hot_threeway,
}

def get_probe_file_name(config, probe_type, probe_prediction, layer):
    model_file_name = os.path.basename(config["model"]).split(".")[0]
    final_output_dir = os.path.join(config["output_dir"], model_file_name)
    return f"{final_output_dir}/resid_{probe_type}_{probe_prediction}_{layer}_.pth"

def get_accuracy_of_othello_gpt(othello_gpt, board_seqs_int, board_seqs_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    total_correct = 0
    total_predictions = 0

    othello_gpt.eval()  # Set the model to evaluation mode

    num_games = 128

    othello_gpt.eval()

    with torch.no_grad():
        for i in range(0, num_games, batch_size):
            batch_end = min(i + batch_size, num_games)
            batch_seqs_int = board_seqs_int[i:batch_end].to(device)
            batch_seqs_string = board_seqs_string[i:batch_end]
            valid_moves = get_valid_moves(batch_seqs_string).to(device) # Batch x 60 x 61

            # Get model predictions
            logits = othello_gpt(batch_seqs_int[:, :-1])
            predictions = logits.argmax(dim=-1) # Batch x 59

            # For every game in the batch, for every move in the game, check if it's valid
            for j in range(predictions.shape[0]):  # Iterate over the games
                for k in range(predictions.shape[1]):
                    # Predictions (0, 0) is the prediction of the correct move AFTER move zero. Where as 
                    # valid_moves (0, 0) is the valid moves FOR move zero. Thus, we need to add one to the valid
                    # moves k index (as it goes to 60) - to check how accurate it is.
                    if valid_moves[j, k + 1, predictions[j, k]] == 1:
                        total_correct += 1
                    total_predictions += 1

    accuracy = total_correct / total_predictions
    return accuracy

def uniform_loss(predictions):
    """
    predictions: Tensor of shape (..., 8, 8, 3) containing logits. 
    Notably, this uses KL divergence to calculate the loss between the predictions and a uniform distribution.
    Which encourages the model to predict uniformly across the three classes.
    """
    # Apply softmax to convert logits to probabilities
    probs = F.softmax(predictions, dim=-1)
    # NOTE: we ignore the first dimension of the tensor, as it's the mode dimension, 
    # which we're not interested in.
    probs = probs[0, ...]
    # Define the uniform distribution (1/3 for each class)
    uniform_dist = torch.full_like(probs, 1 / 3)
    # Calculate the KL divergence loss
    kl_loss = F.kl_div(probs.log(), uniform_dist, reduction='batchmean')
    return kl_loss

def train(config):
    """Train parametric linear probe model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    othello_gpt: HookedTransformer = get_empty_transformer_lens().to(device) # type: ignore
    linear_probe: LinearProbe = LinearProbe(othello_gpt.cfg.d_model, 1, 8, 8, 3).to(device)

    epochs, alpha, lr, wd, valid_every, batch_size, pos_start, pos_end, num_epochs, valid_size, valid_patience, output_dir = (
        config['epochs'], config['alpha'], config["lr"], config["wd"], config["valid_every"], config["batch_size"], config["pos_start"], config["pos_end"], config["num_epochs"], config["valid_size"], config["valid_patience"], config["output_dir"],
    )
    pos_start = 0
    pos_end = othello_gpt.cfg.n_ctx - 0

    probe_optimizer = torch.optim.AdamW(
        linear_probe.parameters(), lr=5e-4 * 3, betas=(0.9, 0.99), weight_decay=wd
    )
    othello_gpt_optimizer = torch.optim.AdamW(
        othello_gpt.parameters(), lr=5e-4, betas=(0.9, 0.99), weight_decay=wd
    )

    one_hot_function = state_stack_to_one_hot_threeway

    # All data
    board_seqs_int, board_seqs_string, board_seqs_int_validation, board_seqs_string_validation = get_board_seqs_int_and_str(config["num_games"])
    valid_indices = torch.arange(valid_size)
    valid_games_int = board_seqs_int_validation[valid_indices]
    valid_games_str = board_seqs_string_validation[valid_indices]
    valid_state_stack = build_state_stack(valid_games_str)
    train_size = board_seqs_int.shape[0]

    # We just pick layer 64 as it's near the start. And so the question is we can we 
    # get the model to _not build_ the world model until later in the layers..
    layer = 4

    print(f"Starting othello_gpt accuracy: {get_accuracy_of_othello_gpt(othello_gpt, board_seqs_int, board_seqs_string)}")
    start_time = int(time.time())

    output_dir = os.path.join('bucket', "gans")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):

        # First, we select a random subset of the data that we're going to 
        # train on. This is equivalent to the random noise generated in a GAN
        train_indices = torch.randperm(train_size)
        train_indices = train_indices[:batch_size]

        # Then, we train the probe. We do this with the same probe training process
        # as before, we we try and train it to detect a mostly linearly relationship 
        # between mine/theirs
        linear_probe.train()

        games_int = board_seqs_int[train_indices].to(device)
        games_str = board_seqs_string[train_indices].to(device)
        state_stack = build_state_stack(games_str).to(device)

        state_stack = state_stack[:, pos_start:pos_end, :, :]

        state_stack_one_hot = one_hot_function(state_stack.to(device))
        with torch.inference_mode():
            _, cache = othello_gpt.run_with_cache(
                games_int.to(device)[:, :-1], return_type=None
            )

            resid_post = cache["resid_post", layer][
                :, pos_start:pos_end
            ]

        # Clone the resid_post tensor to allow gradient computation
        resid_post = resid_post.clone().detach().requires_grad_(True)
        _, train_loss = linear_probe(resid_post, state_stack_one_hot)
        train_loss.backward()

        probe_optimizer.step()
        probe_optimizer.zero_grad()

        # Then, we train the GPT model. We do this with the same GPT training process
        # as before, but we also train it to maximize the entropy of the probe output
        # as well as maximize the likelihood of valid predictions

        othello_gpt.train()
        # NOTE: we don't use inference mode, as we want to autodiff against the cache
        # results for calculating the entropy
        logits, cache = othello_gpt.run_with_cache(
            games_int.to(device)[:, :-1], return_type="logits"
        )
        resid_post = cache["resid_post", layer][
            :, pos_start:pos_end
        ]
        logits_reshaped = logits.view(-1, logits.size(-1))
        targets_reshaped = games_int[:, 1:].reshape(-1)

        # First, let's get the loss for the logits. TODO: do we need the ignore_index
        # TODO: wait, is this right because we need to get the proper targets...
        test_loss = F.cross_entropy(logits_reshaped, targets_reshaped, ignore_index=0)

        # Then, we get the probe output, so we can calculate the entropy of the probe
        probe_out, _ = linear_probe(resid_post, state_stack_one_hot)
        diff_from_uniform_loss = uniform_loss(probe_out)

        # Finally, we calculate the total loss
        total_loss = (alpha * test_loss)**2 + ((1 - alpha) * diff_from_uniform_loss)**2
        #print(f'Total loss: {total_loss}, Test loss: {test_loss}, Probe entropy: {probe_entropy}')
        total_loss.backward()

        othello_gpt_optimizer.step()
        othello_gpt_optimizer.zero_grad()


        if epoch % valid_every == 0:
            print(f"Epoch {epoch} othello_gpt accuracy: {get_accuracy_of_othello_gpt(othello_gpt, board_seqs_int, board_seqs_string)}")
            print(f'Total loss: {total_loss}, Test loss: {test_loss} = {round((alpha * test_loss.item())**2 / total_loss.item() * 100)}%, Probe entropy: {diff_from_uniform_loss} {round(((1 - alpha) * diff_from_uniform_loss.item())**2 / total_loss.item() * 100)}%')
            val_losses = []
            val_accuracies = []
            for val_batch_idx in range(0, valid_size, batch_size):
                _valid_indices = valid_indices[
                    val_batch_idx : val_batch_idx + batch_size
                ]
                _valid_games_int = valid_games_int[_valid_indices]
                _valid_state_stack = valid_state_stack[_valid_indices]
                _valid_state_stack = _valid_state_stack[
                    :, pos_start:pos_end, ...
                ]
                _valid_stack_one_hot = one_hot_function(_valid_state_stack.to(device))

                _val_logits, _val_cache = othello_gpt.run_with_cache(
                    _valid_games_int.to(device)[:, :-1],
                    return_type="logits",
                )
                val_resid_post = _val_cache["resid_post", layer][
                    :, pos_start:pos_end
                ]
                _val_probe_out, val_loss = linear_probe(val_resid_post, _valid_stack_one_hot)

                val_losses.append(val_loss.item() * _valid_indices.shape[0])

                val_preds = _val_probe_out.argmax(-1)
                val_gold = _valid_stack_one_hot.argmax(-1)

                val_results = val_preds == val_gold
                val_accuracy = (
                    val_results.sum() / val_results.numel()
                ).item()
                val_accuracies.append(
                    val_accuracy * _valid_indices.shape[0]
                )

            validation_loss = sum(val_losses) / valid_size
            validation_accuracy = sum(val_accuracies) / valid_size
            print(f"  Probe Validation Accuracy: {validation_accuracy}")
            print(f"  Probe Validation Loss: {validation_loss}")

            # Save the resulting models, under their current time
            # in the bucket/gans/{} folder
            torch.save(othello_gpt.state_dict(), f"{output_dir}/{alpha}_othello_gpt.pth")
            torch.save(linear_probe.state_dict(), f"{output_dir}/{alpha}_linear_probe.pth")
    return start_time

def evaluate(start_time, alpha):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    othello_gpt = get_empty_transformer_lens()
    othello_gpt.load_state_dict(torch.load(f"bucket/gans/{alpha}_othello_gpt.pth"))
    othello_gpt.eval()

    othello_gpt = othello_gpt.to(device)

    linear_probe = LinearProbe(othello_gpt.cfg.d_model, 1, 8, 8, 3)
    linear_probe.load_state_dict(torch.load(f"bucket/gans/{alpha}_linear_probe.pth"))
    linear_probe.eval()
    linear_probe = linear_probe.to(device)

    board_seqs_int, board_seqs_string, board_seqs_int_validation, board_seqs_string_validation = get_board_seqs_int_and_str(10000)

    valid_size = 512
    batch_size = 128
    pos_start = 0
    pos_end = othello_gpt.cfg.n_ctx - 0
    one_hot_function = state_stack_to_one_hot_threeway
    layer = 6

    valid_indices = torch.arange(valid_size)
    valid_games_int = board_seqs_int_validation[valid_indices]
    valid_games_str = board_seqs_string_validation[valid_indices]
    valid_state_stack = build_state_stack(valid_games_str)
    train_size = board_seqs_int.shape[0]

    print(f"Othello_gpt accuracy: {get_accuracy_of_othello_gpt(othello_gpt, board_seqs_int, board_seqs_string)}")
    val_losses = []
    val_accuracies = []
    for val_batch_idx in range(0, valid_size, batch_size):
        _valid_indices = valid_indices[
            val_batch_idx : val_batch_idx + batch_size
        ]
        _valid_games_int = valid_games_int[_valid_indices].to(device)
        _valid_state_stack = valid_state_stack[_valid_indices].to(device)
        _valid_state_stack = _valid_state_stack[
            :, pos_start:pos_end, ...
        ]
        _valid_stack_one_hot = one_hot_function(_valid_state_stack.to(device))

        _val_logits, _val_cache = othello_gpt.run_with_cache(
            _valid_games_int.to(device)[:, :-1],
            return_type="logits",
        )
        val_resid_post = _val_cache["resid_post", layer][
            :, pos_start:pos_end
        ]
        _val_probe_out, val_loss = linear_probe(val_resid_post, _valid_stack_one_hot)

        val_losses.append(val_loss.item() * _valid_indices.shape[0])

        val_preds = _val_probe_out.argmax(-1)
        val_gold = _valid_stack_one_hot.argmax(-1)

        val_results = val_preds == val_gold
        val_accuracy = (
            val_results.sum() / val_results.numel()
        ).item()
        val_accuracies.append(
            val_accuracy * _valid_indices.shape[0]
        )

    validation_loss = sum(val_losses) / valid_size
    validation_accuracy = sum(val_accuracies) / valid_size
    print(f"  Probe Validation Accuracy: {validation_accuracy}")
    print(f"  Probe Validation Loss: {validation_loss}")



if __name__ == "__main__":
    train_config = {
        "lr": 1e-2,
        "wd": 0.01,
        "num_games": 5000000,
        "valid_every": 20,
        "batch_size": 128,
        'epochs': int(1e5),
        "pos_start": 0,
        "pos_end": 0,
        "num_epochs": 1,
        "valid_size": 512,
        "valid_patience": 10,
        "output_dir": "bucket/probes",
    }

    for alpha in [.99, .999, .8, .5, .1, .01]:
        print(f'\n\nRunning on {alpha}:')
        train_config["alpha"] = alpha
        start_time = train(train_config)
        evaluate(start_time, alpha)

# TODO: make it so the probe_prediction and probe_type both save under different file names (same folder, perhaps)
# and then we can read them in and see the predictions.