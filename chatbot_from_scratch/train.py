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
from architecture import Transformer
from dataset_util import ConversationDataset
from config import *

dataset         = ConversationDataset(file_path, max_len = max_len)
transformer     = Transformer(d_model, heads, num_layers, len(dataset.vocab_transform))

loss_fn         = torch.nn.CrossEntropyLoss(ignore_index = PAD_IDX)
optimizer       = torch.optim.AdamW(transformer.parameters(), lr = 0.001, betas = (0.9, 0.98), eps = 1e-9)
scaler          = GradScaler()
transformer.to(device)


# checkpoint  = torch.load('transformer.tar', map_location = device)
# transformer.load_state_dict(checkpoint['model_state_dict'])

def train_epoch():
    losses = 0
    train_dataloader = DataLoader(dataset, batch_size=batch_size,
                                  num_workers=0, pin_memory=True)

    transformer.train()
    counter = 0
    for src, src_mask, tgt_input, tgt_mask, tgt_target in train_dataloader:
        src, src_mask, tgt_input, tgt_mask, tgt_target = src.to(device), src_mask.to(device), tgt_input.to(
            device), tgt_mask.to(device), tgt_target.to(device)
        #print(counter)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = transformer(src, src_mask, tgt_input, tgt_mask)
            pred = pred.flatten(0, 1)
            tgt_target = tgt_target.flatten()

            loss = loss_fn(pred, tgt_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate():
    losses = 0
    accuracy = 0
    total = 0
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(eval_indices))

    transformer.eval()
    for src, src_mask, tgt_input, tgt_mask, tgt_target in eval_dataloader:
        src, src_mask, tgt_input, tgt_mask, tgt_target = src.to(device), src_mask.to(device), tgt_input.to(
            device), tgt_mask.to(device), tgt_target.to(device)

        with torch.no_grad():
            pred = transformer(src, src_mask, tgt_input, tgt_mask)
            pred = pred.flatten(0, 1)
            tgt_target = tgt_target.flatten()

            loss = loss_fn(pred, tgt_target)
            losses += loss.item()

            accuracy += (pred.argmax(-1) == tgt_target).sum().item()
            total += tgt_target.size(0)

    return losses / len(eval_dataloader), accuracy / (total + EPS)



#checkpoint  = torch.load('transformer.tar', map_location = device)
#transformer.load_state_dict(checkpoint['model_state_dict'])
print('----------')
transformer = transformer.to(device)

for epoch in range(1, epochs + 1):
    print(f"Start epoch: {epoch}")

    start_time  = timer()
    train_loss  = train_epoch()
    end_time    = timer()

    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    if(epoch % 2 == 0):
        state = { 'model_state_dict': transformer.state_dict(), 'optimizer_state_dict': optimizer.state_dict() }
        torch.save(state, f'transformer_{epoch}.tar')

print('finish training')
