
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import *

def infer(inp):
    inp = "<startofstring> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    model.config.max_length = X.shape[1] + 256
    output = model.generate(X, attention_mask=a )
    output = tokenizer.decode(output[0])
    return output
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>",
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load("weights/model_state.pt"))
model = model.to(device)
model.eval()