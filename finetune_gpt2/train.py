
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tqdm
from config import *
from dataset_util import ChatData
from torch.utils.data import DataLoader
from torch.optim import Adam

def train(chatData, model, optim, device):
    for i in tqdm.tqdm(range(epochs)):
        counter=0
        model.train()
        total_loss = 0
        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            total_loss+=loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")
        if(i%30 == 0):
            torch.save(model.state_dict(), f"model_state{i}.pt")
        model.eval()
        print("loss :",total_loss/len(chatData))
        print(infer("Merhaba dizlerimde de dayanılmaz bir ağrı var. Kaslarım ağrıyor ve kemiğimde iğne batar gibi bir ağrı var. Bunun için ne yapabilirim?"))
def infer(inp):
    inp = "<startofstring> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    model.config.max_length = X.shape[1] + 256
    output = model.generate(X, attention_mask=a )
    output = tokenizer.decode(output[0])
    return output
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>",
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])



chatData = ChatData("tum_data.csv", tokenizer)
chatData =  DataLoader(chatData, batch_size=15)


model = model.to(device)
model.train()


optim = Adam(model.parameters(), lr=1e-3)

print("training .... ")
train(chatData, model, optim,device)
