# Adapted from: https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Othello_GPT.ipynb#scrollTo=2lkORR6SSIof
# NOTE: there are some bugs. Namely, dropout is weird, and gets lost. But I think this could be fine
# The main thing we want to check is that the model mostly works - this is something that we can verify when
# in a bit. Let's try and get the probes setup first. 

import torch
from mingpt.model import GPT, GPTConfig
from transformer_lens import HookedTransformerConfig, HookedTransformer, utils
import einops

def get_model_mingpt_state_dict(checkpoint_path):
    # Load the saved checkpoint
    state_dict = torch.load(checkpoint_path)
    return state_dict

def get_model_mingpt(checkpoint_path):
    # Initialize the GPT model with the same configuration as when we created it
    mconf = GPTConfig(
        vocab_size=61,
        block_size=59,
        n_layer=8,
        n_head=8,
        n_embd=512
    )
    model = GPT(mconf)

    state_dict = get_model_mingpt_state_dict(checkpoint_path)

    model.load_state_dict(state_dict)
    return model

def get_model_transformer_lens(checkpoint_path):
    # Initialize the Transformer model with the same configuration as when we created it
    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)

    mingpt_state_dict = get_model_mingpt_state_dict(checkpoint_path)
    transformer_lens_state_dict = _convert_to_transformer_lens_format(mingpt_state_dict)

    model.load_and_process_state_dict(transformer_lens_state_dict)
    return model


def _convert_to_transformer_lens_format(in_sd, n_layers=8, n_heads=8):
    out_sd = {}
    out_sd["pos_embed.W_pos"] = in_sd["pos_emb"].squeeze(0)
    out_sd["embed.W_E"] = in_sd["tok_emb.weight"]

    out_sd["ln_final.w"] = in_sd["ln_f.weight"]
    out_sd["ln_final.b"] = in_sd["ln_f.bias"]
    out_sd["unembed.W_U"] = in_sd["head.weight"].T

    for layer in range(n_layers):
        out_sd[f"blocks.{layer}.ln1.w"] = in_sd[f"blocks.{layer}.ln1.weight"]
        out_sd[f"blocks.{layer}.ln1.b"] = in_sd[f"blocks.{layer}.ln1.bias"]
        out_sd[f"blocks.{layer}.ln2.w"] = in_sd[f"blocks.{layer}.ln2.weight"]
        out_sd[f"blocks.{layer}.ln2.b"] = in_sd[f"blocks.{layer}.ln2.bias"]

        out_sd[f"blocks.{layer}.attn.W_Q"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.query.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_Q"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.query.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_K"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.key.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_K"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.key.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_V"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.value.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_V"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.value.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_O"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.proj.weight"],
            "d_model (head d_head) -> head d_head d_model",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_O"] = in_sd[f"blocks.{layer}.attn.proj.bias"]

        out_sd[f"blocks.{layer}.mlp.b_in"] = in_sd[f"blocks.{layer}.mlp.0.bias"]
        out_sd[f"blocks.{layer}.mlp.W_in"] = in_sd[f"blocks.{layer}.mlp.0.weight"].T
        out_sd[f"blocks.{layer}.mlp.b_out"] = in_sd[f"blocks.{layer}.mlp.2.bias"]
        out_sd[f"blocks.{layer}.mlp.W_out"] = in_sd[f"blocks.{layer}.mlp.2.weight"].T

    return out_sd


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "./bucket/checkpoints/gpt_at_20241015_215935.ckpt"  # Update with actual path
    model1 = get_model_mingpt(checkpoint_path).to(device)
    model2 = get_model_transformer_lens(checkpoint_path).to(device)

    # An example input
    sample_input = torch.tensor(
        [
            [
                20, 19, 18, 10, 2, 1, 27, 3, 41, 42, 34, 12, 4, 40, 11, 29, 43, 13, 48, 56, 33, 39, 22, 44, 24, 5, 46, 6, 32, 36, 51, 58, 52, 60, 21, 53, 26, 31, 37, 9, 25, 38, 23, 50, 45, 17, 47, 28, 35, 30, 54, 16, 59, 49, 57, 14, 15, 55, 7,
            ] 
        ]
    ).to(device)
    # The argmax of the output (ie the most likely next move from each position)
    sample_output = torch.tensor(
        [
            [
                21, 41, 40, 34, 40, 41, 3, 11, 21, 43, 40, 21, 28, 50, 33, 50, 33, 5, 33, 5, 52, 46, 14, 46, 14, 47, 38, 57, 36, 50, 38, 15, 28, 26, 28, 59, 50, 28, 14, 28, 28, 28, 28, 45, 28, 35, 15, 14, 30, 59, 49, 59, 15, 15, 14, 15, 8, 7, 8,
            ]
        ]
    ).to(device)

    print(model1(sample_input)[0].argmax(dim=-1) == model2(sample_input).argmax(dim=-1))
