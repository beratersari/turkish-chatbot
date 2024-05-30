import torch
data_path ="../tum_data.csv"
epochs = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")