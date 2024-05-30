import spacy
import pandas as pd
import math
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import GradScaler
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from config import special_symbols, UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-1 * torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding           = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2]  = torch.sin(pos * den)
        pos_embedding[:, 1::2]  = torch.cos(pos * den)
        pos_embedding           = pos_embedding.unsqueeze(-2)

        self.dropout        = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.dropout(tokens + self.pos_embedding[:tokens.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()

        self.embedding  = nn.Embedding(vocab_size, emb_size)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long())

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k    = d_model // heads
        self.heads  = heads

        self.dropout    = nn.Dropout(0.1)
        self.query      = nn.Linear(d_model, d_model)
        self.key        = nn.Linear(d_model, d_model)
        self.value      = nn.Linear(d_model, d_model)
        self.out        = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        query   = self.query(query)
        key     = self.key(key)
        value   = self.value(value)

        query   = query.view(query.shape[0], self.heads, -1, self.d_k)
        key     = key.view(key.shape[0], self.heads, -1, self.d_k)
        value   = value.view(value.shape[0], self.heads, -1, self.d_k)

        scores      = torch.matmul(query, key.transpose(2, 3))
        scores      = scores / math.sqrt(query.size(-1))

        if mask is not None:
            min_type_value  = torch.finfo(scores.dtype).min
            scores  = scores.masked_fill(mask == 0, min_type_value)

        weights     = F.softmax(scores, dim = -1)
        weights     = self.dropout(weights)

        context     = torch.matmul(weights, value)
        context     = context.transpose(1, 2).flatten(2)

        interacted  = self.out(context)
        return interacted

class FeedForward(nn.Module):
    def __init__(self, d_model: int, middle_dim: int = 2048):
        super(FeedForward, self).__init__()

        self.fc1        = nn.Linear(d_model, middle_dim)
        self.fc2        = nn.Linear(middle_dim, d_model)
        self.dropout    = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super(EncoderLayer, self).__init__()

        self.layernorm      = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward   = FeedForward(d_model)
        self.dropout        = nn.Dropout(0.1)

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        interacted          = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted          = self.layernorm(interacted + embeddings)
        feed_forward_out    = self.dropout(self.feed_forward(interacted))
        encoded             = self.layernorm(feed_forward_out + interacted)
        return encoded

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super(DecoderLayer, self).__init__()

        self.layernorm      = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead  = MultiHeadAttention(heads, d_model)
        self.feed_forward   = FeedForward(d_model)
        self.dropout        = nn.Dropout(0.1)

    def forward(self, embeddings: Tensor, encoded: Tensor, target_mask: Tensor) -> Tensor:
        query               = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, target_mask))
        query               = self.layernorm(query + embeddings)
        interacted          = self.dropout(self.src_multihead(query, encoded, encoded, None))
        interacted          = self.layernorm(interacted + query)
        feed_forward_out    = self.dropout(self.feed_forward(interacted))
        decoded             = self.layernorm(feed_forward_out + interacted)
        return decoded

class Transformer(nn.Module):
    def __init__(self, d_model: int, heads: int, num_layers: int, vocab_size: int, dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.d_model        = d_model
        self.vocab_size     = vocab_size
        self.src_tok_emb    = TokenEmbedding(self.vocab_size, d_model)
        self.tgt_tok_emb    = TokenEmbedding(self.vocab_size, d_model)
        self.pos_encoding   = PositionalEncoding(d_model, dropout = dropout)
        self.encoder        = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder        = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        self.logit          = nn.Linear(d_model, self.vocab_size)

    def encode(self, src_words: Tensor, src_mask: Tensor) -> Tensor:
        src_embeddings = self.pos_encoding(self.src_tok_emb(src_words))
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
        return src_embeddings

    def decode(self, target_words: Tensor, target_mask: Tensor, src_embeddings: Tensor) -> Tensor:
        tgt_embeddings = self.pos_encoding(self.tgt_tok_emb(target_words))

        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, target_mask)

        out = self.logit(tgt_embeddings)
        return out

    def forward(self, src_words: Tensor, src_mask: Tensor, target_words: Tensor, target_mask: Tensor) -> Tensor:
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded)
        return decoded

class ConversationDataset(Dataset):
    def __init__(self, file_path, max_len = 100, init_dataset = True) -> None:
        super().__init__()

        if init_dataset:
            self._init_dataset(file_path, max_len)

    def __len__(self):
        return len(self.src_batch)

    def __getitem__(self, idx):
        src, tgt    = self.src_batch[idx], self.tgt_batch[idx]

        tgt_input   = tgt[:-1]
        tgt_target  = tgt[1:]

        src_mask, tgt_mask          = self.create_src_mask(src), self.create_tgt_mask(tgt_input)
        return src, src_mask, tgt_input, tgt_mask, tgt_target

    def _init_dataset(self, file_path, max_len):
        print('read csv')
        df = pd.read_csv(file_path, sep='|')
        df = df.astype(str)

        print('convert to list')
        question    = df['question'].to_list()
        answer      = df['answer'].to_list()
        words       = question + answer

        print('build vocab & transform')
        self.token_transform    = get_tokenizer('spacy', language='en_core_web_sm')
        self.vocab_transform    = build_vocab_from_iterator(self.yield_tokens(words), min_freq = 1, specials = special_symbols, special_first = True)
        self.text_transform     = self.sequential_transforms(self.token_transform, self.vocab_transform, self.tensor_transform)

        print('set default index')
        self.vocab_transform.set_default_index(UNK_IDX)

        print('transform batch')
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in zip(question, answer):
            src_batch.append(self.text_transform(src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform(tgt_sample.rstrip("\n")))

        print('pad sequence')
        src_batch   = pad_sequence(src_batch, padding_value = PAD_IDX, batch_first = True)
        tgt_batch   = pad_sequence(tgt_batch, padding_value = PAD_IDX, batch_first = True)

        print('clip the length to fit data')
        self.src_batch  = src_batch[:, :max_len]
        self.tgt_batch  = tgt_batch[:, :max_len]

        print('finish init dataset')

    def yield_tokens(self, data_iter):
        for data_sample in data_iter:
            yield self.token_transform(data_sample)

    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    def tensor_transform(self, token_ids):
        return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones((sz, sz))).transpose(0, 1)
        return mask

    def create_src_mask(self, src):
        src_seq_len         = src.shape[-1]
        src_lookahead_mask  = torch.ones((src_seq_len, src_seq_len)).bool()
        src_padding_mask    = (src != PAD_IDX)
        src_mask            = src_padding_mask.unsqueeze(0) & src_lookahead_mask

        return src_mask.unsqueeze(0)

    def create_tgt_mask(self, tgt):
        tgt_seq_len         = tgt.shape[-1]
        tgt_lookahead_mask  = self.generate_square_subsequent_mask(tgt_seq_len).bool()
        tgt_padding_mask    = (tgt != PAD_IDX)
        tgt_mask            = tgt_padding_mask.unsqueeze(0) & tgt_lookahead_mask

        return tgt_mask.unsqueeze(0)