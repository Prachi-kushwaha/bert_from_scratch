from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 5,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'squad',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

# class Config:

    # dataset
    train_path = "data/train.txt"
    max_seq_length = 512

    # tokenizer
    vocab_size = 30522
    lower_case = True

    # model
    hidden_size = 768
    num_layers = 12
    num_heads = 12

    # training
    batch_size = 32
    lr = 3e-4
    epochs = 10

    # MLM
    mlm_probability = 0.15

    # system
    seed = 42