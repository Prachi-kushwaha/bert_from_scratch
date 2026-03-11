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

class ModelConfig:

    def __init__(self):
        self.batch_size = 8
        self.seq_len = 34
        self.d_model = 512
        self.vocab_size = 1000
        self.segment_vocab_size = 2
        self.max_position_embeddings = 105
        self.dropout = 0.1
        self.dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        self.d_ff = 2048
        self.h = 8
        self.num_hidden_layers = 6
