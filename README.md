# Sarcasm Detection in News Headlines

NLP project comparing traditional token-based classification with embedding-based approaches for detecting sarcasm in news headlines.

## Overview

This notebook demonstrates two approaches to sarcasm detection:

1. **Baseline**: Logistic regression on tokenized sequences
2. **Enhanced**: Logistic regression on learned word embeddings

## Dataset

Using the Sarcasm Headlines Dataset from Kaggle containing news headlines labeled as sarcastic or not.

## Methodology

### Preprocessing

- Text normalization (lowercase, punctuation removal)
- Tokenization using NLTK
- Stop word filtering
- Vocabulary building with top 5000 words

### Models

**Baseline Model**

- Input: Padded token indices
- Algorithm: Logistic Regression
- Features: Raw token IDs

**Embedding Model**

- Input: Same padded sequences
- Transformation: 64-dimensional embeddings
- Aggregation: Average pooling across sequence
- Algorithm: Logistic Regression on embeddings

## Key Findings

The notebook compares:

- Classification accuracy
- ROC AUC scores
- Confusion matrices
- Model performance on custom examples

## Usage

```python
# Install dependencies
pip install kagglehub nltk scikit-learn tensorflow pandas numpy matplotlib

# Run notebook
jupyter notebook sarcasm_detection.ipynb
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- NLTK
- pandas, numpy, matplotlib
- kagglehub

## Structure

1. Dataset acquisition and loading
2. Exploratory analysis
3. Text preprocessing
4. Tokenization and vectorization
5. Baseline model training
6. Embedding layer creation
7. Enhanced model training
8. Performance comparison
9. Custom headline testing

## Results

The notebook provides:

- Detailed metrics for both approaches
- Visual comparison of model performance
- ROC curve analysis
- Interactive prediction function

## Educational Goals

- Understanding text preprocessing pipelines
- Token-based vs embedding-based representations
- Binary classification for NLP tasks
- Model evaluation and comparison

---

_Project for NLP coursework_
