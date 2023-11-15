# Arabic_Text_Summerization_With_Pytorch
# ICTMCV2 Code Documentation

## Overview
The provided code is a Jupyter Notebook (`ICTMCV2.ipynb`) for a sequence-to-sequence (Seq2Seq) model trained for text summarization tasks in Arabic. It uses the Hugging Face `transformers` library and leverages pre-trained models for Arabic language processing.

## Dependencies
Make sure you have the necessary dependencies installed. You can install them using the following commands:
```python
!pip install transformers
!pip install datasets
!pip install arabert
!pip install sentencepiece
!pip install arabicnlp
```

## Preprocessing
The code defines several preprocessing functions to clean and prepare the input text for the summarization model. These functions include removing diacritics, extra spaces, repeated characters, quotes, punctuation, and certain special characters. The `Sequential` function is used to apply a sequence of these preprocessing functions to the input text.

## Model Initialization
The script initializes and loads a Seq2Seq model for Arabic text summarization using the `transformers` library. The model can be configured to use different pre-trained models such as BERT, BART, etc. The selected model is loaded, and a tokenizer is created.

## Data Loading
The code loads labeled validation data (`labeled_validation_dataset.jsonl`) and additional datasets required for training and evaluation.

## Training
The training process involves creating a custom PyTorch dataset (`SummarizationDataset`) for the training data. The `collate_fn` function is used to collate batches for training. The script defines functions for training epochs, calculating loss, and evaluating accuracy.

## Model Fine-Tuning
The code fine-tunes the loaded model on the training dataset and evaluates its performance on the validation dataset. The training process involves optimizing the model parameters using the AdamW optimizer.

## LoRA Layer
The script implements a custom LoRA (Learnable Rank Adaptation) layer, which is a modification to the original model's layers. The LoRA layer is used to adapt the model's weights based on their rank.

## Inference
The code provides functions for generating summaries using the trained model. It uses the `generate` function to produce summaries for input text.

## Evaluation
The script evaluates the model's performance on the validation dataset, calculating Rouge scores and cosine similarity scores between generated summaries and ground truth summaries.

## Results and Analysis
The final section of the code presents an analysis of the generated summaries, including Rouge scores, cosine similarity scores, and comparisons between generated and ground truth summaries.

## Model Saving
The trained model and its parameters can be saved using the `save_pretrained` method.

## Conclusion
This code serves as a comprehensive solution for training, fine-tuning, and evaluating a Seq2Seq model for Arabic text summarization using pre-trained transformer models.
