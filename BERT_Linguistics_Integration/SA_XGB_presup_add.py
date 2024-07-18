import xgboost as xgb
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

def load_embeddings(path):
    data = torch.load(path)
    features = data.to(device)
    return features.cpu().numpy()  

def load_labels(path):
    data = torch.load(path)
    features, labels = data['embeddings'].to(device), data['labels'].to(device)
    return labels.cpu().numpy() 

def load_all_embeddings(directory, num_files, mode):
    all_embeddings, all_labels = [], []
    for i in range(1, num_files + 1):
        path = os.path.join(directory, f'embeddings_{i}.pt')
        embeddings = load_embeddings(path)
        if mode == "train":
            naive_path = os.path.join("/Users/ishaansingh/Documents/CS224N_FinalProject/sent_train", f'batch_{i}.pt')
        else:
            naive_path = os.path.join("/Users/ishaansingh/Documents/CS224N_FinalProject/sent_test", f'batch_{i}.pt')
        labels = load_labels(naive_path)
        all_embeddings.append(embeddings)
        all_labels.append(labels)
    return np.vstack(all_embeddings), np.concatenate(all_labels)

train_embeddings, train_labels = load_all_embeddings('/Users/ishaansingh/Documents/CS224N_FinalProject/SA_embeddings_add_train', 85, "train")
test_embeddings, test_labels = load_all_embeddings('/Users/ishaansingh/Documents/CS224N_FinalProject/SA_embeddings_add_test', 10, "test")

dtrain = xgb.DMatrix(train_embeddings, label=train_labels)
dtest = xgb.DMatrix(test_embeddings, label=test_labels)

params = {
    'objective': 'binary:logistic',  # binary classification
    'max_depth': 6,  
    'learning_rate': 0.15,  
    'n_estimators': 100, 
    'eval_metric': 'logloss', 
}

model = xgb.train(params, dtrain, num_boost_round=10, evals=[(dtrain, 'train'), (dtest, 'test')])

predictions = model.predict(dtest)
predicted_labels = np.round(predictions) 

accuracy = accuracy_score(test_labels, predicted_labels)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')
