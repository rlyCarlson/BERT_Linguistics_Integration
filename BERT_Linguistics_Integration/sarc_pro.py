import pandas as pd
import json

data = pd.read_csv('/Users/ishaansingh/Downloads/train-balanced.csv', delimiter='|', header=None, names=['thread', 'responses', 'labels'])
with open('/Users/ishaansingh/Downloads/comments.json', 'r') as file:
    comments = json.load(file)
test_data = pd.read_csv('/Users/ishaansingh/Downloads/test-balanced.csv', delimiter='|', header=None, names=['thread', 'responses', 'labels'])
with open('/Users/ishaansingh/Downloads/comments.json', 'r') as file:
    comments = json.load(file)
def parse_ids(id_string):
    return id_string.split()

def get_comment_text(comment_id):
    return comments.get(comment_id, {}).get('text', '').strip('"')

data['thread'] = data['thread'].apply(parse_ids)
data['responses'] = data['responses'].apply(parse_ids)
data['labels'] = data['labels'].apply(lambda x: list(map(int, parse_ids(x))))
test_data['thread'] = test_data['thread'].apply(parse_ids)
test_data['responses'] = test_data['responses'].apply(parse_ids)
test_data['labels'] = test_data['labels'].apply(lambda x: list(map(int, parse_ids(x))))
data['thread_text'] = data['thread'].apply(lambda ids: [get_comment_text(comment_id) for comment_id in ids])
data['response_text'] = data['responses'].apply(lambda ids: [get_comment_text(response_id) for response_id in ids])
test_data['thread_text'] = test_data['thread'].apply(lambda ids: [get_comment_text(comment_id) for comment_id in ids])
test_data['response_text'] = test_data['responses'].apply(lambda ids: [get_comment_text(response_id) for response_id in ids])
print(data.head())
print("Number of rows:", data.shape[0])
responses_df = pd.DataFrame({
    'response_text': pd.Series([item for sublist in data['response_text'] for item in sublist]),
    'label': pd.Series([item for sublist in data['labels'] for item in sublist])
})
responses_df['response_text'] = responses_df['response_text'].str.strip().str.strip('"')
print(responses_df.head())
csv_output_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/responses_SARC.csv' 
responses_df.to_csv(csv_output_path, index=False)  

print(f"DataFrame has been saved to {csv_output_path}.")

print(test_data.head())
print("Number of rows:", test_data.shape[0])
test_responses_df = pd.DataFrame({
    'response_text': pd.Series([item for sublist in test_data['response_text'] for item in sublist]),
    'label': pd.Series([item for sublist in test_data['labels'] for item in sublist])
})
test_responses_df['response_text'] = test_responses_df['response_text'].str.strip().str.strip('"')
print(test_responses_df.head())
csv_output_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/test_responses_SARC.csv'
test_responses_df.to_csv(csv_output_path, index=False)

print(f"DataFrame has been saved to {csv_output_path}.")