import torch

from configuration import ModelConfig
from dataset import get_data, build_dataloader
from tokenizer import BertTokenizer
from model import BertForQA
from trainer import Trainer


def main():

    config = ModelConfig()

    # 1️⃣ Load dataset
    data = get_data(config)

    # 2️⃣ Train tokenizer
    tokenizer = BertTokenizer.train_tokenizer(data)

    # 3️⃣ Build dataloader
    loader = build_dataloader(data, tokenizer)

    # 4️⃣ Build model
    model = BertForQA(config)

    # 5️⃣ Trainer
    trainer = Trainer(model, loader)

    # 6️⃣ Training loop
    for epoch in range(3):

        loss = trainer.train_epoch()

        print("epoch:", epoch, "loss:", loss)


if __name__ == "__main__":
    main()