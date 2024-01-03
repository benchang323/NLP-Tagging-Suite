# NLP Tagging Suite

## Overview

NLP Tagging Suite is a robust tagging framework designed for Natural Language Processing tasks. It utilizes Hidden Markov Models (HMM) and Conditional Random Fields (CRF) to perform accurate and efficient tagging of text. This suite is ideal for linguistic analysis, data categorization, and various NLP applications.

## Features

- **Hidden Markov Models (HMM):** Implements HMM for sequential data processing in language tasks.
- **Conditional Random Fields (CRF):** Advanced tagging with RNN-based features for enhanced accuracy.
- **Data Handling:** Capable of processing both supervised and unsupervised data sets.
- **Performance Optimization:** Fine-tuned for efficient processing of large datasets.
- **Command-Line Interface:** Easy-to-use CLI for model training, evaluation, and tagging.
- **Numerical Stability:** Designed to handle small probability products, ensuring consistent performance.

## Files

- `hmm.py`: Implementation of the Hidden Markov Model.
- `crf.py`: Implementation of Conditional Random Fields with RNN features.
- `eval.py`: Script for tagging accuracy assessment.
- `corpus.py`, `lexicon.py`: Manage corpus data and word attribute construction.
- `tag.py`: Command-line tool for model management.
- `test_ic.py`, `test_en.py`: Test scripts for specific datasets.
- Additional utility scripts like `integerize.py` and `logsumexp_safe.py`.

## Tech Stack

- **Programming Language:** Python
- **Core Library:** PyTorch for tensor operations and model building.

## Libraries/Dependencies

- `PyTorch`: For model architecture and tensor operations.
- `Pathlib`, `Tqdm`: For efficient file handling and progress indication.
- Standard Python libraries for data manipulation and computation.

## Installation
```
git clone https://github.com/benchang323/NLP-Tagging-Suite.git
cd NLP-Tagging-Suite
```
