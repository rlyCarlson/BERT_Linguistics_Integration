from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased').to(device)

def generate_embeddings(text):
    if not isinstance(text, str): 
        return None  
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}  # Move input to the MPS device
    with torch.no_grad():
        outputs = bert_model(**encoded_input)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
    return hidden_states.squeeze()

file_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/responses_SARC.csv'
test_file_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/test_responses_SARC.csv'
SARC_df = pd.read_csv(file_path)
test_SARC_df = pd.read_csv(test_file_path)

SARC_df['response_text'] = SARC_df['response_text'].astype(str)
test_SARC_df['response_text'] = test_SARC_df['response_text'].astype(str)

half_index = len(SARC_df)
subset_df = SARC_df.iloc[:half_index] 
test_subset_df = test_SARC_df

tqdm.pandas(desc="Generating Embeddings")
subset_df['embeddings'] = subset_df['response_text'].progress_apply(lambda text: generate_embeddings(text))
test_subset_df['embeddings'] = test_subset_df['response_text'].progress_apply(lambda text: generate_embeddings(text))

subset_df.dropna(subset=['embeddings'], inplace=True)
test_subset_df.dropna(subset=['embeddings'], inplace=True)

features = torch.stack(subset_df['embeddings'].tolist()).to(device)
labels = torch.tensor(subset_df['label'].values).to(device)
train_dataset = TensorDataset(features, labels)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

test_features = torch.stack(test_subset_df['embeddings'].tolist()).to(device)
test_labels = torch.tensor(test_subset_df['label'].values).to(device)
test_dataset = TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = TextClassifier(input_size=768, hidden_size=128, num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
training_losses = [] 

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    training_losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Average Loss: {avg_loss}')

plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test data: {accuracy}%')
