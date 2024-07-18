import torch
from transformers import BertTokenizer, BertModel
import csv
from tqdm import tqdm
import itertools

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()  

def generate_embeddings(sentence1, sentence2):
    if not isinstance(sentence1, str) or not isinstance(sentence2, str):
        return None
    text = sentence1 + " [SEP] " + sentence2
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        outputs = bert_model(**encoded_input)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
    return hidden_states.squeeze()

def read_tuples_from_csv_and_generate_embeddings(file_path):
    embeddings = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for sentence1, sentence2 in tqdm(itertools.islice(reader, 999), desc="Processing CSV File"):
            try:
                if sentence1 and sentence2:
                    emb = generate_embeddings(sentence1, sentence2)
                    if emb is not None:
                        embeddings.append(emb)
            except Exception as e:
                print(f"Error processing row: {sentence1}, {sentence2}. Error: {e}")
        return torch.stack(embeddings) if embeddings else None

file_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/snli_combined.csv'
embeddings_tensor = read_tuples_from_csv_and_generate_embeddings(file_path)

if embeddings_tensor is not None:
    torch.save(embeddings_tensor, 'snli_presup_train_embeddings.pt')
    print("Embeddings saved.")
else:
    print("No embeddings generated.")
