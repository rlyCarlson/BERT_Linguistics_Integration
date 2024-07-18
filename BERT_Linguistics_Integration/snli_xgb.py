import xgboost as xgb
import torch
import numpy as np

def load_data_xgb(embeddings_file, labels_file):
    embeddings = torch.load(embeddings_file, map_location=torch.device('cpu')).numpy()
    labels = torch.load(labels_file, map_location=torch.device('cpu')).numpy()
    return xgb.DMatrix(embeddings, label=labels)


train_data = load_data_xgb('/Users/ishaansingh/Documents/CS224N_FinalProject/snli_embed/snli_train_embeddings.pt', '/Users/ishaansingh/Documents/CS224N_FinalProject/snli_embed/snli_train_labels.pt')
test_data = load_data_xgb('/Users/ishaansingh/Documents/CS224N_FinalProject/snli_embed/snli_test_embeddings.pt', '/Users/ishaansingh/Documents/CS224N_FinalProject/snli_embed/snli_test_labels.pt')

params = {
    'objective': 'multi:softmax',  
    'num_class': 3,                
    'max_depth': 6,                
    'eta': 0.3,                    
    'eval_metric': 'mlogloss',     
    'verbosity': 1                 
}

num_rounds = 100 
bst = xgb.train(params, train_data, num_rounds, evals=[(test_data, 'test')])

predictions = bst.predict(test_data)
labels = test_data.get_label()
accuracy = np.sum(predictions == labels) / len(labels)
print(f'Accuracy: {accuracy:.4f}')
