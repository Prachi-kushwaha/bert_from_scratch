import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.segment_embeddings = nn.Embedding(config.segment_vocab_size, config.d_model)

        self.layerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_layer_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )

        self.register_buffer(
            "segment_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False
        )

    def forward(
        self,
        input_ids=None,
        segment_ids=None,
        position_ids=None,
        input_embeds=None
    ):

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = input_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        # word embeddings
        if input_embeds is None:
            input_embeds = self.word_embeddings(input_ids)

        # segment ids default
        if segment_ids is None:
            segment_ids = self.segment_ids[:, :seq_length]

        # position ids default
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            position_ids = position_ids.expand(batch_size,seq_length)

        segment_embedding = self.segment_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeds + segment_embedding + position_embeddings

        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, config)->None:
        super().__init__()
        self.d_model = config.d_model
        self.h = config.h
        assert self.d_model % self.h == 0, "d_model is not divisible by h"

        self.d_k = self.d_model // self.h
        self.dropout = nn.Dropout(config.dropout)
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)

        self.w_o = nn.Linear(config.d_model, config.d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is None:
            attention_scores = dropout(attention_scores)

        return (attention_scores@value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)






