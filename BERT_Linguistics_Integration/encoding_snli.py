import torch
from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm  

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def generate_embeddings(sentence1, sentence2):
    if not isinstance(sentence1, str) or not isinstance(sentence2, str):
        return None
    text = sentence1 + " [SEP] " + sentence2 # notice use of separation token
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        outputs = bert_model(**encoded_input)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
    return hidden_states.squeeze()

def read_jsonl_file_and_generate_embeddings(file_path):
    with open(file_path, 'r') as file:
        total_lines = sum(1 for line in file)

    embeddings = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Processing SNLI Dataset"):
            data = json.loads(line)
            if data['gold_label'] != '-': 
                emb = generate_embeddings(data['sentence1'], data['sentence2'])
                if emb is not None:
                    embeddings.append(emb)
    return torch.stack(embeddings)

file_path = '/Users/ishaansingh/Downloads/snli_1.0/snli_1.0_test.jsonl'
embeddings_tensor = read_jsonl_file_and_generate_embeddings(file_path)

torch.save(embeddings_tensor, 'snli_test_embeddings.pt')
print("Embeddings saved.")
