import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 512
heads = 8
num_layers = 6
epochs= 300
EPS = 1e-9
batch_size = 16
max_len = 256
train_len = 140000
file_path = '../tum_data.csv'
train = True
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3