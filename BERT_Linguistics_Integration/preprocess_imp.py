import json
import os
from sklearn.model_selection import train_test_split

directory = 'IMPPRES/implicature/'

data = []

for filename in os.listdir(directory):
    print(filename)
    if filename.endswith('.jsonl'):
        filepath = os.path.join(directory, filename)
        print(f'Reading file: {filepath}')

        with open(filepath, 'r') as file:
            c = 0
            for line in file:
                try:
                    json_obj = json.loads(line)
                    if c == 0: print(json_obj.keys())
                    c += 1

                    if "spec_relation" in json_obj.keys() and (json_obj['spec_relation'] == "implicature_PtoN" or json_obj['spec_relation'] == "implicature_NtoP"):
                        data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}\nError: {e}")

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=42)

output_directory = "split_data"
os.makedirs(output_directory, exist_ok=True)

with open(os.path.join(output_directory, "train_implicature_2.jsonl"), 'w') as train_file:
    for entry in train_data:
        json_line = json.dumps(entry)
        train_file.write(json_line + '\n')

with open(os.path.join(output_directory, "val_implicature_2.jsonl"), 'w') as val_file:
    for entry in val_data:
        json_line = json.dumps(entry)
        val_file.write(json_line + '\n')

with open(os.path.join(output_directory, "test_implicature_2.jsonl"), 'w') as test_file:
    for entry in test_data:
        json_line = json.dumps(entry)
        test_file.write(json_line + '\n')

print("Data split into train, validation, and test sets.")
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(val_data)}")
print(f"Number of test examples: {len(test_data)}")
print(len(train_data) + len(val_data) + len(test_data))
