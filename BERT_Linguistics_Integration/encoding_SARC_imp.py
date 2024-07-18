from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
from tqdm import tqdm
import os 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased').to(device)

def generate_embeddings(text):
    if not isinstance(text, str):
        return None
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        outputs = bert_model(**encoded_input)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
    return hidden_states.squeeze()

file_path = "/Users/ishaansingh/Documents/CS224N_FinalProject/imp_SARC_test.csv"
SARC_df = pd.read_csv(file_path)
SARC_df['implicature'] = SARC_df['implicature'].astype(str)

tqdm.pandas(desc="Generating Implicature Embeddings")

base_dir = '/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_imp_Embeddings_test'
batch_size = 10000
for i in tqdm(range(0, len(SARC_df), batch_size), desc="Processing Batches"):
    batch_df = SARC_df.iloc[i:i+batch_size]
    batch_df['embeddings'] = batch_df['implicature'].progress_apply(lambda text: generate_embeddings(text))
    batch_df.dropna(subset=['embeddings'], inplace=True)  

    # new directory for each batch
    batch_dir = os.path.join(base_dir, f'Batch_{i//batch_size + 1}')
    os.makedirs(batch_dir, exist_ok=True)  

    # .pt file in the new directory
    features = torch.stack(batch_df['embeddings'].tolist()).to(device)
    torch.save({'features': features},
               os.path.join(batch_dir, 'embeddings.pt'))

    print(f"Embeddings and labels for Batch {i//batch_size + 1} saved successfully in {batch_dir}.")