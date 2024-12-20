# make deterministic
from mingpt.utils import set_seed
set_seed(44)

import os
import math
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.dataset import CharDataset
from data import get_test_and_train_datasets
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
import wandb


def train(model: GPT, train_dataset: CharDataset, test_dataset: CharDataset):

    max_epochs = 250
    # initialize a trainer instance and kick off training
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    tconf = TrainerConfig(
        max_epochs=max_epochs, 
        batch_size=512 * 8,  # assuming 8 GPU's
        learning_rate=5e-4,
        lr_decay=True, 
        warmup_tokens=len(train_dataset)*train_dataset.block_size*5, 
        final_tokens=len(train_dataset)*train_dataset.block_size*max_epochs,
        num_workers=0, 
        ckpt_path=f"./bucket/checkpoints/gpt_at{t_start}.ckpt", 
    )
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    device = trainer.device

    # Start a new run
    wandb.init(project='othello', name=f"gpt_at{t_start}")

    trainer.train()

if __name__ == '__main__':

    train_dataset, test_dataset = get_test_and_train_datasets(total=25 * 1000000) # 
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)

    train(model, train_dataset, test_dataset)