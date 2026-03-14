import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from configuration import Config

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
        self.writer = SummaryWriter("runs/bert_qa")
        self.global_step = 0

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

        
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)

            self.global_step += 1

        return total_loss / len(self.dataloader)

    def save_checkpoint(self, epoch, path="weights"):
        os.makedirs(path, exist_ok=True)

        save_path = os.path.join(path, f"bert_qa_epoch_{epoch}.pt")

        torch.save({
                "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, save_path)

        print(f"Model saved at {save_path}")