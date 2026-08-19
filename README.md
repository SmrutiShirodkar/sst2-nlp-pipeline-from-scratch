# sst2-nlp-pipeline-from-scratch

A complete NLP pipeline built from first principles, from raw text to a trained and evaluated sentiment
classifier, on the Stanford Sentiment Treebank (SST-2) binary movie review sentiment dataset.

## Problem

Given a movie review as free text, predict whether the sentiment is positive or negative. The interesting part
of this project is not the final classifier itself, a logistic regression is about as simple as classifiers get,
but everything that has to happen before a model can see the text as numbers at all: tokenization, normalization,
stopword removal, stemming, vocabulary construction, and feature extraction. I built each of these components
myself rather than relying on a single library call, specifically to understand what each step actually buys you
and where it falls short.

## Approach

The pipeline runs in four stages:

**Text preprocessing.** I implemented and compared a naive whitespace tokenizer against NLTK's `word_tokenize`,
lowercasing, punctuation removal, stopword removal, and a custom suffix-stripping stemmer compared against
NLTK's Porter and Lancaster stemmers. Each function has its own sanity checks run against hand-constructed test
cases, which is how I caught, for example, that a naive whitespace tokenizer fails on contractions like `isn't`
while NLTK's tokenizer correctly splits it into `is` and `n't`.

**Feature extraction.** Preprocessed reviews are converted into bag-of-words feature vectors: a fixed vocabulary
is built from the training set, and every document becomes a vector of word counts indexed against that
vocabulary.

**Model.** A logistic regression classifier implemented in PyTorch from the ground up, including a custom
`Dataset` and `DataLoader`, explicit weight initialization, binary cross-entropy loss, and an Adam optimizer,
with the forward pass, loss computation, and backward pass all wired together manually rather than through a
higher-level training API. This was deliberate: the goal was to understand each moving part of a training loop,
not just call `.fit()`.

**Evaluation.** The trained model is evaluated on a held-out test split. Beyond overall accuracy, I added a
precision, recall, F1, and confusion matrix breakdown, since the dataset has a roughly 45/55 negative/positive
class split rather than an even one, which makes accuracy alone an incomplete picture of performance.

## Key design decisions

- **Bag-of-words over pretrained embeddings.** This was a deliberate choice to keep the feature representation
  interpretable. A bag-of-words logistic regression lets every model weight be read directly as "how much does
  this specific word push the prediction toward positive or negative," which embeddings and more complex
  architectures would obscure.
- **Building preprocessing components from scratch rather than only using library defaults.** Comparing my own
  whitespace tokenizer and stemmer against NLTK's implementations directly, side by side, made the tradeoffs of
  each concrete rather than assumed.
- **Precision, recall, F1, and a confusion matrix in addition to accuracy.** Given the near-even but not
  perfectly balanced class split in this dataset, accuracy alone would not surface whether the model's errors
  are concentrated in false positives or false negatives.

## Results and what they mean

The model reaches roughly 80 percent test accuracy, against a roughly 50 percent random-guessing baseline
(stated explicitly in the notebook rather than left implicit), a meaningful and clearly stated improvement for a
first, deliberately simple model. Precision, recall, F1, and a confusion matrix are computed on the same held-out
test set to show where the model's errors actually concentrate, rather than relying on accuracy alone to tell
the whole story. Exact metric values are visible in the notebook's own output since they depend on the specific
training run.

The closing section interprets the trained model directly: the words with the largest positive weights
(`terrific`, `refreshing`, `thoughtprovoking`) and the largest negative weights (`worst`, `devoid`, `failur`,
the stemmed form of "failure") line up with human intuition about what predicts positive or negative sentiment,
which is a useful sanity check that the model learned something real rather than an artifact of the data.

## Limitations

- Bag-of-words discards word order entirely. A review like "not good" and "good" would share the word "good" as
  a feature with no representation of the negation, which a sequence model (RNN, or a pretrained transformer
  encoder) would capture and a bag-of-words model cannot.
- The `evaluate` function defaults to `device="cuda"` throughout the notebook, a carryover from the environment
  it was originally developed in. Running this notebook on a CPU-only machine requires passing `device="cpu"`
  explicitly wherever `evaluate` or the training loop is called.
- No hyperparameter tuning (learning rate, vocabulary size cutoff, batch size) was performed; the values used
  are reasonable defaults rather than the result of a search.
- No comparison against a stronger baseline (TF-IDF weighting instead of raw counts, or a pretrained embedding
  model) is included, which would be the natural next step to contextualize how much headroom is left above this
  approach.

## Setup

Download the SST-2 dataset from [here](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip) and unzip it so that
`train.tsv` and `dev.tsv` sit inside a `data/SST-2/` folder relative to the notebook.

## Skills demonstrated

**NLP Fundamentals:** tokenization (custom and NLTK), stemming (custom Porter/Lancaster-style comparison),
stopword removal, vocabulary construction, bag-of-words feature extraction

**Deep Learning from First Principles:** PyTorch `Dataset`/`DataLoader`, manual forward pass and backward pass
wiring, explicit weight initialization, binary cross-entropy loss, Adam optimizer, built without a high-level
training API to demonstrate understanding of each component

**Evaluation:** precision, recall, F1, confusion matrix analysis on an imbalanced-class dataset, baseline
comparison against random guessing, model interpretability via learned-weight inspection

**Engineering Practice:** hand-constructed unit tests for preprocessing functions, side-by-side comparison of
custom implementations against established library behavior to validate design choices, explicit documentation
of unaddressed limitations (no hyperparameter search, no stronger baseline, hardcoded device default)
