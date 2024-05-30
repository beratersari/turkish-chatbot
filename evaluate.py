from finetune_gpt2.predict import infer
from mistral7b.chat import mistral_chatbot

import pandas as pd
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
# BERT-based semantic similarity
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def calculate_f1(ground_truth, model_answers):
    gt_embeddings = [get_sentence_embedding(sentence) for sentence in ground_truth]
    model_embeddings = [get_sentence_embedding(sentence) for sentence in model_answers]

    similarities = [cosine_similarity(gt, pred) for gt, pred in zip(gt_embeddings, model_embeddings)]
    # Consider a threshold for determining similarity-based "correct" answers
    threshold = 0.5
    binary_predictions = [1 if sim >= threshold else 0 for sim in similarities]
    binary_ground_truth = [1] * len(ground_truth)

    return f1_score(binary_ground_truth, binary_predictions)


data_df = pd.read_csv("C:\\Users\\BERAT\\chatbot\\tum_data.csv")
data_df.dropna(inplace=True)
ground_truth = data_df["Response_tr"].tolist()
gpt2_answers = []
mistral_answers = []
for i in range(len(data_df)):
    question = data_df.iloc[i]["Context_tr"]
    ans = mistral_chatbot(question)
    mistral_answers.append(ans)

for i in range(len(data_df)):
    question = data_df.iloc[i]["Context_tr"]
    ans = infer(question)
    ans = ans.split("<bot>:")[1]
    # remove the leading space after endofstring token
    index = ans.find("<endofstring>")
    if (index != -1):
        ans = ans[:index]
    gpt2_answers.append(ans)

f1_gpt2 = calculate_f1(data_df["Response_tr"].tolist(), gpt2_answers)
print(f"F1 Score for GPT-2: {f1_gpt2}")

f1_mistral = calculate_f1(data_df["Response_tr"].tolist(), mistral_answers)
print(f"F1 Score for Mistral: {f1_mistral}")