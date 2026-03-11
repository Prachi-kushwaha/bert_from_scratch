import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration import ModelConfig

config = ModelConfig()

class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.segment_embeddings = nn.Embedding(config.segment_vocab_size, config.d_model)

        self.layerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

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
        self.dropout = nn.Dropout(config.dropout_prob)
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
        if dropout is not None:
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


class ResidualConnection(nn.Module):

    def __init__(self,features:int, config) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_prob)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForwardBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.linear_2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        #(Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) --> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))

class EncoderBlock(nn.Module):

    def __init__(self,features:int, self_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, config) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,config) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x , lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderBlock(
                config.d_model,
                MultiHeadAttentionBlock(config),
                FeedForwardBlock(config),
                config
            )
            for _ in range(config.num_hidden_layers)]
        )

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class BertForQA(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embeddings = InputEmbeddings(config)
        self.encoder = Encoder(config)

        self.qa_outputs = nn.Linear(config.d_model, 2)

    def forward(self, input_ids, attention_mask):

        x = self.embeddings(input_ids)

        mask = attention_mask.unsqueeze(1).unsqueeze(2)

        x = self.encoder(x, mask)

        logits = self.qa_outputs(x)

        start_logits, end_logits = logits.split(1, dim=-1)

        return start_logits.squeeze(-1), end_logits.squeeze(-1)