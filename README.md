# Integrating Extra Linguistic Meaning into the BERT Framework

## Overview
In this project, we aim to improve BERT's performance on advanced Natural Language Inference (NLI) tasks by integrating additional linguistic meanings that humans naturally interpret from the input text. 

## Methodology
We propose a method that generates presuppositions and implicatures from the input text and combines them with BERT's encodings using three different combination techniques:
- Concatenation
- Adding
- SVD Decomposition

## Linguistic Meanings Generation
We finetuned the T5 language model on the IMPPRES dataset to generate these extra linguistic meanings.

## Evaluation
We evaluate our model on three downstream tasks that benefit from more nuanced linguistic meanings:
- Inference Classification
- Irony/Sarcasm Detection
- Sentiment Analysis

## Results
Downstream task evaluation shows modest improvements with the additive and concatenation techniques when incorporating the encodings from the presuppositions and implicatures. These results reveal the need for a more robust generation process.

## Conclusion
By integrating additional linguistic meanings into BERT's encodings, we aim to enhance the model's performance on complex NLI tasks, pushing the boundaries of what NLP models can achieve in understanding and interpreting human language.
