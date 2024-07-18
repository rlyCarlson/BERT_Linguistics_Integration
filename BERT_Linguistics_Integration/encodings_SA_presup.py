from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
from tqdm import tqdm
import os  

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased').to(device)

df = pd.read_csv('/Users/ishaansingh/Documents/CS224N_FinalProject/SA_presuppositions_test.csv')
df.fillna('', inplace=True)

print(df.head())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased').to(device)
bert_model.eval()

output_dir = "/Users/ishaansingh/Documents/CS224N_FinalProject/SA_presup_embed_test"
os.makedirs(output_dir, exist_ok=True)

batch_size = 100
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    texts = batch['presupposition'].tolist()
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = bert_model(**encoded_input)
        embeddings = outputs.last_hidden_state

    torch.save(embeddings, os.path.join(output_dir, f'embeddings_{i//batch_size}.pt'))
