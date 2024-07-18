import os
import json
from datasets import Dataset
from datasets import load_dataset, load_metric
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset
from evaluate import load
import numpy as np
import gc

train_dataset_path = "split_data/train_implicature.jsonl"
val_dataset_path = "split_data/val_implicature.jsonl"
test_dataset_path = "split_data/test_implicature.jsonl"

model_name = "t5-base-finetuned-implicatures-small-3"
base_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(base_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")
model.to(device)
print(device)

with open(train_dataset_path, 'r') as file:
    train_data = [json.loads(line) for line in file]

with open(val_dataset_path, 'r') as file:
    validation_data = [json.loads(line) for line in file]

with open(test_dataset_path, 'r') as file:
    test_data = [json.loads(line) for line in file]


train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.rename_column("sentence1", "input")
train_dataset = train_dataset.rename_column("sentence2", "output")

val_dataset = Dataset.from_list(validation_data)
val_dataset = val_dataset.rename_column("sentence1", "input")
val_dataset = val_dataset.rename_column("sentence2", "output")

test_dataset = Dataset.from_list(test_data)
test_dataset = test_dataset.rename_column("sentence1", "input")
test_dataset = test_dataset.rename_column("sentence2", "output")

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return {k: v for k, v in model_inputs.items()}

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)

tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

bleu = load("bleu")
rouge = load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    preds = torch.argmax(torch.tensor(preds), dim=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    print("BLEU score:", bleu_score)
    print("ROUGE scores:", rouge_scores)
    return {"bleu": bleu_score["bleu"], "rouge": rouge_scores}

eval_args = TrainingArguments(
    output_dir="evaluation",
    per_device_eval_batch_size=1,
    no_cuda = True,
)

eval_trainer = Trainer(
    model=model,
    args=eval_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

chunk_size = 100

total_samples = len(tokenized_train_dataset)

combined_bleu_scores = []
combined_rouge_scores = []

for i in range(0, total_samples, chunk_size):
    val_chunk = tokenized_train_dataset.select(range(i, min(i + chunk_size, total_samples)))

    with torch.no_grad():
        chunk_metrics = eval_trainer.evaluate(val_chunk, metric_key_prefix=f"val_chunk_{i}")
    print(chunk_metrics)
    combined_bleu_scores.append(chunk_metrics[f"val_chunk_{i}_bleu"])
    combined_rouge_scores.append(chunk_metrics[f'val_chunk_{i}_rouge'])
    gc.collect()
    torch.cuda.empty_cache()

average_bleu_score = np.mean(combined_bleu_scores)
average_rouge_scores = {}
for key in combined_rouge_scores[0].keys():
    average_rouge_scores[key] = np.mean([score[key] for score in combined_rouge_scores])
print("Train Metrics (Combined):")
print("BLEU Score:", average_bleu_score)
print("ROUGE Scores:", average_rouge_scores)