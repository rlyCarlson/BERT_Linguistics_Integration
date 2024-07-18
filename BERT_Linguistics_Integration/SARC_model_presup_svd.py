import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn, optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

def load_embeddings(path):
    data = torch.load(path)
    return data['embeddings'].to(device), data['labels'].to(device)

def load_all_embeddings(base_path, num_batches):
    all_embeddings = []
    all_labels = []
    for i in range(1, num_batches + 1):
        path = f'{base_path}/Batch_{i}/embeddings.pt'
        embeddings, labels = load_embeddings(path)
        all_embeddings.append(embeddings)
        all_labels.append(labels)
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

train_embeddings, train_labels = load_all_embeddings('/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_presup_embeddings_svd_train', 26)
test_embeddings, test_labels = load_all_embeddings('/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_presup_embeddings_svd_test', 7)

batch_size = 1000
train_dataset = TensorDataset(train_embeddings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_embeddings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

model = TextClassifier(input_size=768*2, hidden_size=128, num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 125
training_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    train_loop = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')
    for data, target in train_loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        train_loop.set_postfix(loss=loss.item())
    training_losses.append(sum(epoch_losses) / len(epoch_losses))

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
validation_losses = []
test_loop = tqdm(test_loader, desc='Testing')
with torch.no_grad():
    for data, target in test_loop:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, target)
        validation_losses.append(loss.item())
        total += target.size(0)
        correct += (predicted == target).sum().item()
        test_loop.set_postfix(accuracy=f'{100 * correct / total:.2f}%', loss=f'{loss.item():.4f}')

accuracy = 100 * correct / total
avg_validation_loss = sum(validation_losses) / len(validation_losses)
print(f'Accuracy on test data: {accuracy:.2f}%, Average Validation Loss: {avg_validation_loss:.4f}')
