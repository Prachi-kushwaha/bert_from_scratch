import torch

from configuration import Config
from dataset import get_data, build_dataloader
from tokenizer import BertTokenizer
from model import BertForQA
from trainer import Trainer


def main():

    config = Config()

    # Load dataset
    data = get_data(config.datasource)

    # Train tokenizer
    tokenizer = BertTokenizer.train_tokenizer(data)

    # Build dataloader
    loader = build_dataloader(data, tokenizer)

    # Build model
    model = BertForQA(config)

    # Trainer
    trainer = Trainer(model, loader)

    # Training loop
    for epoch in range(3):

        loss = trainer.train_epoch()

        print("epoch:", epoch, "loss:", loss)
        trainer.save_checkpoint(epoch)


if __name__ == "__main__":
    main()