import json
from tqdm import tqdm 
import torch

def save_labels_from_jsonl(file_path, output_file):
    labels = []
    with open(file_path, 'r') as file:
        total_lines = sum(1 for line in file)
    
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Collecting Labels"):
            data = json.loads(line)
            if data['gold_label'] != '-':
                labels.append(data['gold_label'])
    
    label_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    labels_tensor = torch.tensor([label_mapping[label] for label in labels])

    torch.save(labels_tensor, output_file)
    print(f"Labels saved to {output_file}.")

file_path = '/Users/ishaansingh/Downloads/snli_1.0/snli_1.0_test.jsonl'
output_file = 'snli_test_labels.pt'
save_labels_from_jsonl(file_path, output_file)
