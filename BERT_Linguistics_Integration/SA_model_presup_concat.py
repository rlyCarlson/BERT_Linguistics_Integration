import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn, optim
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

def load_embeddings(path):
    data = torch.load(path)
    features = data.to(device)
    return features

def load_labels(path):
    data = torch.load(path)
    features, labels = data['embeddings'].to(device), data['labels'].to(device)
    return labels

def load_all_embeddings(directory, num_files, mode):
    all_embeddings, all_labels = [], []
    max_length = 0
    for i in range(1, num_files + 1):
        path = os.path.join(directory, f'embeddings_{i}.pt')
        embeddings = load_embeddings(path)
        max_length = max(max_length, embeddings.size(-1))
    for i in range(1, num_files + 1):
        path = os.path.join(directory, f'embeddings_{i}.pt')
        if mode == "train":
            naive_path = os.path.join("/Users/ishaansingh/Documents/CS224N_FinalProject/sent_train", f'batch_{i}.pt')
        else:
            naive_path = os.path.join("/Users/ishaansingh/Documents/CS224N_FinalProject/sent_test", f'batch_{i}.pt')
        embeddings = load_embeddings(path)
        labels = load_labels(naive_path)
        all_embeddings.append(embeddings)
        all_labels.append(labels)
    return torch.cat(all_embeddings), torch.cat(all_labels)

train_embeddings, train_labels = load_all_embeddings('/Users/ishaansingh/Documents/CS224N_FinalProject/SA_embeddings_cat_train', 85, "train")
test_embeddings, test_labels = load_all_embeddings('/Users/ishaansingh/Documents/CS224N_FinalProject/SA_embeddings_cat_test', 10, "test")

print("train_embeddings shape:", train_embeddings.shape)
print("train_labels shape:", train_labels.shape)

train_loader = DataLoader(TensorDataset(train_embeddings, train_labels), batch_size=1000, shuffle=True)
test_loader = DataLoader(TensorDataset(test_embeddings, test_labels), batch_size=1000)

class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = TextClassifier(input_size=1536, hidden_size=128, num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

training_losses = []
for epoch in range(100):
    model.train()
    epoch_losses = []
    for data, target in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/100'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    training_losses.append(sum(epoch_losses) / len(epoch_losses))

plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, target in tqdm(test_loader, desc='Testing'):
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        ind, predicted = torch.max(outputs.data, -1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test data: {accuracy:.2f}%')
