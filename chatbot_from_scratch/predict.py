from dataset_util import ConversationDataset
from architecture import Transformer
from config import *
import torch

dataset         = ConversationDataset(file_path, max_len = max_len)
checkpoint  = torch.load('transformer.tar')
device= 'cpu'
transformer     = Transformer(d_model, heads, num_layers, len(dataset.vocab_transform))
transformer.load_state_dict(checkpoint['model_state_dict'])
transformer.to(device)

def predict_answer(question):
    src = dataset.text_transform(question).unsqueeze(0).to(device)  # Move to device here
    src_mask = dataset.create_src_mask(src).to(device)  # Move to device here
    src = src

    encoded = transformer.encode(src, src_mask)
    start_y = torch.tensor([[BOS_IDX]]).type_as(src).to(device)  # Move to device here
    ys = start_y

    for i in range(max_len - 1):
        tgt_mask = dataset.create_tgt_mask(ys).to(device)  # Move to device here
        logit = transformer.decode(ys, tgt_mask, encoded)
        next_word = logit.argmax(-1)

        ys = torch.cat([start_y, next_word], dim=1)
        if next_word[0, -1].item() == EOS_IDX:
            break

    ys = ys.flatten().cpu().tolist()  # Move to CPU for processing
    ys = " ".join(dataset.vocab_transform.lookup_tokens(ys)).replace("<bos>", "").replace("<eos>", "")
    return ys
