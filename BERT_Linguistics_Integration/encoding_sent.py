import os
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train_data = dataset['test']

class RottenTomatoesDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {'text': item['text'], 'label': item['label']}

train_dataset = RottenTomatoesDataset(train_data)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

save_directory = 'sent_test'
os.makedirs(save_directory, exist_ok=True)

def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    encoded_dict = encode_texts(texts)
    encoded_dict = {key: val.to(device) for key, val in encoded_dict.items()}
    return texts, encoded_dict, torch.tensor(labels).to(device)

data_loader = DataLoader(train_dataset, batch_size=100, collate_fn=collate_fn)

bert_model.eval()
with torch.no_grad():
    for idx, (texts, batch, labels) in enumerate(tqdm(data_loader, desc="Processing")):
        outputs = bert_model(**batch)
        embeddings = outputs.last_hidden_state
        torch.save({'embeddings': embeddings.cpu(), 'labels': labels.cpu(), 'texts': texts}, 
                   os.path.join(save_directory, f'batch_{idx}.pt'))

print("Embeddings, labels, and texts are saved successfully.")
