from torch.utils.data import Dataset
import pandas as pd
import random
class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        data_df = pd.read_csv(path)
        data_df["text"] ="a"
        data_df.dropna(inplace=True)
        self.X = data_df["Context_tr"].tolist()
        self.Y =  data_df["Response_tr"].tolist()
        for idx, i in enumerate(self.X):
            self.X[idx] = "<startofstring> "+self.X[idx]+" <bot>: "+self.Y[idx]+" <endofstring>"

        self.X = self.X[:-1]
        random.shuffle(self.X)
        print(len(self.X))

        print(self.X[0])

        self.X_encoded = tokenizer(self.X,max_length=256, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])