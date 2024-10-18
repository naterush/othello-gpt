import json
import random
import os
import numpy as np
import torch
import torch.nn as nn

import einops
from fancy_einsum import einsum
from torcheval.metrics.functional import multiclass_f1_score
import torch.nn.functional as F
from tqdm import tqdm

from transformer_lens_adapters import get_model_transformer_lens

from othello_utils import (
    state_stack_to_one_hot_threeway_black_white,
    state_stack_to_one_hot_threeway,
    build_state_stack,
)

from data import get_board_seqs_int_and_str

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

def train(config):
    """Train parametric linear probe model."""
    print("Training config:")
    print(json.dumps(config, indent=4))
    othello_gpt = get_model_transformer_lens(config["model"])

    lr = config["lr"]
    wd = config["wd"]
    rows = config["rows"]
    cols = config["cols"]
    valid_every = config["valid_every"]
    batch_size = config["batch_size"]
    pos_start = config["pos_start"]
    pos_end = othello_gpt.cfg.n_ctx - config["pos_end"]
    num_epochs = config["num_epochs"]
    valid_size = config["valid_size"]
    valid_patience = config["valid_patience"]
    output_dir = config["output_dir"]

    model_file_name = os.path.basename(config["model"]).split(".")[0]
    final_output_dir = os.path.join(output_dir, model_file_name)

    assert os.path.isdir(output_dir)
    if not os.path.isdir(final_output_dir):
        os.makedirs(final_output_dir)

    board_seqs_int, board_seqs_string, board_seqs_int_validation, board_seqs_string_validation = get_board_seqs_int_and_str(config["num_games"])

    valid_indices = torch.arange(valid_size)
    valid_games_int = board_seqs_int_validation[valid_indices]
    valid_games_str = board_seqs_string_validation[valid_indices]
    valid_state_stack = build_state_stack(valid_games_str)
    train_size = board_seqs_int.shape[0]

    modes = 1
    options = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for probe_type, ProbeClass in PROBE_TYPE_TO_CLASS.items():
        for probe_prediction, one_hot_function in ONE_HOT_TYPE_TO_FUNCTION.items():
            print(f"Training {probe_type} probe to predict {probe_prediction}")
            for layer in tqdm(range(8)):
                print(f"Training layer {layer}!")
                done_training = False
                lowest_val_loss = float('inf')

                probe_model = ProbeClass(othello_gpt.cfg.d_model, modes, rows, cols, options).to(device)
                optimiser = torch.optim.AdamW(
                    probe_model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=wd
                )

                torch.manual_seed(42)

                train_seen = 0
                for epoch in range(num_epochs):
                    if done_training:
                        print(f"Training seen: {train_seen}")
                        break

                    full_train_indices = torch.randperm(train_size)
                    for idx in tqdm(range(0, train_size, batch_size)):
                        if done_training:
                            print(f"Training seen: {train_seen}")
                            break
                        train_seen += batch_size
                        indices = full_train_indices[idx : idx + batch_size]
                        games_int = board_seqs_int[indices]
                        games_str = board_seqs_string[indices]
                        state_stack = build_state_stack(games_str)

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
                        _, train_loss = probe_model(resid_post, state_stack_one_hot)
                        train_loss.backward()

                        optimiser.step()
                        optimiser.zero_grad()

                        if idx % valid_every == 0:
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
                                _val_probe_out, val_loss = probe_model(val_resid_post, _valid_stack_one_hot)

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
                            print(f"  Validation Accuracy: {validation_accuracy}")
                            print(f"  Validation Loss: {validation_loss}")
                            if validation_loss < lowest_val_loss:
                                print(f"  New lowest valid loss! {validation_loss}")
                                curr_patience = 0
                                torch.save(
                                    probe_model.state_dict(), get_probe_file_name(config, probe_type, probe_prediction, layer)
                                )

                                lowest_val_loss = validation_loss

                            else:
                                curr_patience += 1
                                print(
                                    f"  Did not beat previous best ({lowest_val_loss})"
                                )
                                print(f"  Current patience: {curr_patience}")
                                if curr_patience >= valid_patience:
                                    print("  Ran out of patience! Stopping training.")
                                    done_training = True

def evaluate(config):
    """
    Evaluate parametric linear probe model.
    """
    othello_gpt = get_model_transformer_lens(config["model"])
    board_seqs_int, board_seqs_string, _, _ = get_board_seqs_int_and_str(config["num_games"])

    test_size = 1000
    board_seqs_int = board_seqs_int[-test_size:]
    board_seqs_string = board_seqs_string[-test_size:]

    games_int = board_seqs_int
    games_str = board_seqs_string
    all_indices = torch.arange(test_size)
    batch_size = 128
    orig_state_stack = build_state_stack(games_str)

    pos_start = 0
    pos_end = othello_gpt.cfg.n_ctx - 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = 1
    options = 3

    for probe_type, ProbeClass in PROBE_TYPE_TO_CLASS.items():
        for probe_prediction, one_hot_function in ONE_HOT_TYPE_TO_FUNCTION.items():
            for layer in range(8):
                probe_model = ProbeClass(othello_gpt.cfg.d_model, modes, config["rows"], config["cols"], options).to(device)
                probe_model.load_state_dict(torch.load(get_probe_file_name(config, probe_type, probe_prediction, layer)))
                probe_model.eval()

                accs = []
                per_timestep_num_correct = torch.zeros((59, 8, 8))
                all_preds = []
                all_groundtruths = []
                for idx in range(0, test_size, batch_size):
                    indices = all_indices[idx : idx + batch_size]
                    _games_int = games_int[indices]

                    state_stack = orig_state_stack[
                        indices, pos_start:pos_end, :, :
                    ]
                    state_stack_one_hot = one_hot_function(state_stack.to(device))

                    logits, cache = othello_gpt.run_with_cache(
                        _games_int.to(device)[:, :-1], return_type="logits"
                    )
                    resid_post = cache["resid_post", layer][
                        :, pos_start:pos_end
                    ]
                    probe_out, _ = probe_model(resid_post, state_stack_one_hot)

                    preds = probe_out.argmax(-1)
                    groundtruth = state_stack_one_hot.argmax(-1)
                    test_results = preds == groundtruth
                    test_acc = (test_results.sum() / test_results.numel()).item()
                    per_timestep_num_correct += test_results[0].sum(0).cpu()
                    all_preds.append(preds)
                    all_groundtruths.append(groundtruth)
                    accs.append(test_acc * indices.shape[0])

                _all_preds = torch.cat(all_preds, dim=1)
                _all_gt = torch.cat(all_groundtruths, dim=1)
                f1_score = multiclass_f1_score(
                    _all_preds.view(-1), _all_gt.view(-1), num_classes=3
                )
                print(f"{probe_type} predicting {probe_prediction}: Layer {layer} F1_score: {f1_score}")

if __name__ == "__main__":
    train_config = {
        "model": "bucket/checkpoints/gpt_at_20241016_154216.ckpt",
        "lr": 1e-2,
        "wd": 0.01,
        "rows": 8,
        "cols": 8,
        "num_games": 100000,
        "valid_every": 200,
        "batch_size": 128,
        "pos_start": 0,
        "pos_end": 0,
        "num_epochs": 1,
        "valid_size": 512,
        "valid_patience": 10,
        "output_dir": "bucket/probes",
    }

    train(train_config)
    evaluate(train_config)

# TODO: make it so the probe_prediction and probe_type both save under different file names (same folder, perhaps)
# and then we can read them in and see the predictions.