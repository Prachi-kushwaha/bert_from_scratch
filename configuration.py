from pathlib import Path

class Config:

    def __init__(self):

        # training
        self.batch_size = 8
        self.num_epochs = 5
        self.lr = 1e-4

        # model
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

        # dataset
        self.datasource = "squad"

        # paths
        self.model_folder = "weights"
        self.model_basename = "tmodel_"
        self.preload = None
        self.tokenizer_file = "tokenizer_{0}.json"
        self.experiment_name = "runs/tmodel"

        # create path object
        self.model_folder_path = Path(self.model_folder)



def get_weights_file_path(config, epoch: str):
    model_folder = Path(f"{config['datasource']}_{config['model_folder']}")
    model_folder.mkdir(parents=True, exist_ok=True)

    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])