import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch

device = torch.device("cpu")

def load_embeddings(path):
    data = torch.load(path, map_location=device)
    return data['features'].numpy(), data['labels'].numpy()

def load_all_embeddings(base_path, num_batches):
    all_embeddings = []
    all_labels = []
    for i in range(1, num_batches + 1):
        path = f'{base_path}/Batch_{i}/embeddings.pt'
        embeddings, labels = load_embeddings(path)
        all_embeddings.append(embeddings)
        all_labels.append(labels)
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

train_embeddings, train_labels = load_all_embeddings('/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_Embeddings', 26)
test_embeddings, test_labels = load_embeddings('/Users/ishaansingh/Documents/CS224N_FinalProject/test_embeddings.pt')

clf = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
clf.fit(train_embeddings, train_labels)

train_predictions = clf.predict(train_embeddings)
test_predictions = clf.predict(test_embeddings)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

xgb.plot_importance(clf)
plt.title('Feature Importance')
plt.show()
