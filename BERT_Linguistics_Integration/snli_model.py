import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm  

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  

def load_data(embeddings_file, labels_file):
    embeddings = torch.load(embeddings_file, map_location=device)
    labels = torch.load(labels_file, map_location=device)
    dataset = TensorDataset(embeddings, labels)
    return DataLoader(dataset, batch_size=64, shuffle=True)

train_loader = load_data('/Users/ishaansingh/Documents/CS224N_FinalProject/snli_embed/snli_train_embeddings.pt', '/Users/ishaansingh/Documents/CS224N_FinalProject/snli_embed/snli_train_labels.pt')
test_loader = load_data('/Users/ishaansingh/Documents/CS224N_FinalProject/snli_embed/snli_test_embeddings.pt', '/Users/ishaansingh/Documents/CS224N_FinalProject/snli_embed/snli_test_labels.pt')

input_size = 768 
num_classes = 3 
model = SimpleNN(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss() # built in softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader)
