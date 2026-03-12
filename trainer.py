import torch
import torchmetrics
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from dataset import SquadDataset
writer = SummaryWriter()


class Trainer:

    def __init__(self, model, dataloader, lr=3e-5, device="cuda"):

        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for batch in self.dataloader:

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)

            start_logits, end_logits = self.model(input_ids, attention_mask)

            start_loss = self.loss_fn(start_logits, start_positions)
            end_loss = self.loss_fn(end_logits, end_positions)

            loss = (start_loss + end_loss) / 2

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)