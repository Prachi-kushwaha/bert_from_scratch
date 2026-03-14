import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def get_data(datasource):
    ds_raw = load_dataset(datasource, split='train[:10]')

    data = []
    for item in ds_raw:
        context = item["context"]
        question = item["question"]
        answer = item["answer"]["text"][0]

        data.append({
            "context":context,
            "question":question,
            "answer":answer
        })

    return data

class SquadDataset(Dataset):

    def __init__(self, data, tokenizer, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        context = item["context"]
        question = item["question"]
        answer_text = item["answer"]
        answer_start = context.find(answer_text)

        # encode question + context
        encoding = self.tokenizer._tokenizer.encode(
            question,
            context
        )

        input_ids = encoding.ids
        attention_mask = encoding.attention_mask
        offsets = encoding.offsets

        start_token = 0
        end_token = 0

        answer_end = answer_start + len(answer_text)

        # find token positions
        for i, (start, end) in enumerate(offsets):

            if start <= answer_start < end:
                start_token = i

            if start < answer_end <= end:
                end_token = i
                break

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "start_positions": torch.tensor(start_token),
            "end_positions": torch.tensor(end_token),
        }


def build_dataloader(data, tokenizer):

    dataset = SquadDataset(data, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    return loader

def collate_fn(batch):

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    input_ids = nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    attention_mask = nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0
    )

    start_positions = torch.tensor([item["start_positions"] for item in batch])
    end_positions = torch.tensor([item["end_positions"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_positions,
        "end_positions": end_positions
    }