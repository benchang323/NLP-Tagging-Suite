#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Module for constructing a lexicon of word attributes.

from collections import Counter
import logging, numpy as np, re
from pathlib import Path
from typing import Optional, Set

import torch

from corpus import TaggedCorpus, BOS_WORD, EOS_WORD, OOV_WORD, Word

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def build_lexicon(corpus: TaggedCorpus,
                  one_hot: bool = False,
                  embeddings_file: Optional[Path] = None,
                  log_counts: bool = False,
                  affixes: bool = True,
                  shape: bool = False) -> torch.Tensor:
    """Returns a lexicon, implemented as a matrix Tensor
    where each row defines real-valued attributes for one of
    the words in corpus.vocab.  This is a wrapper method that
    horizontally concatenates 0 or more matrices that provide 
    different kinds of attributes."""

    matrices = [torch.empty(len(corpus.vocab), 0)]  # start with no features for each word

    if one_hot: matrices.append(one_hot_lexicon(corpus))
    if embeddings_file is not None: matrices.append(embeddings_lexicon(corpus, embeddings_file))
    if log_counts: matrices.append(log_counts_lexicon(corpus))
    # if affixes: matrices.append(affixes_lexicon(corpus))
    # if shape: matrices.append(word_shape_lexicon(corpus))

    return torch.cat(matrices, dim=1)   # horizontally concatenate 

def one_hot_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a matrix with as many rows as corpus.vocab, where 
    each row specifies a one-hot embedding of the corresponding word.
    This allows us to learn features that are specific to the word."""

    return torch.eye(len(corpus.vocab))  # identity matrix

def embeddings_lexicon(corpus: TaggedCorpus, file: Path) -> torch.Tensor:
    """Return a matrix with as many rows as corpus.vocab, where 
    each row specifies a vector embedding of the corresponding word.
    
    The second argument is a lexicon file in the format of Homework 2 and 3, 
    which is used to look up the word embeddings.

    The lexicon entries BOS, EOS, OOV, and OOL will be treated appropriately
    if present.  In particular, any words that are not in the lexicon
    will get the embedding of OOL (or 0 if there is no such embedding).
    """

    vocab = corpus.vocab
    with open(file) as f:
        filerows, cols = [int(i) for i in next(f).split()]   # first line gives num of rows and cols
        matrix = torch.empty(len(vocab), cols)   # uninitialized matrix
        seen: Set[int] = set()                   # the words we've found embeddings for
        ool_vector = torch.zeros(cols)           # use this for other words if there is no OOL entry
        specials = {'BOS': BOS_WORD, 'EOS': EOS_WORD, 'OOV': OOV_WORD}

        # Run through the words in the lexicon, keeping those that are in the vocab.
        for line in f:
            first, *rest = line.strip().split("\t")
            word = Word(first)
            vector = torch.tensor([float(v) for v in rest])
            assert len(vector) == cols     # check that the file didn't lie about # of cols

            if word == 'OOL':
                assert word not in vocab   # make sure there's not an actual word "OOL"
                ool_vector = vector
            else:
                if word in specials:    # map the special word names that may appear in lexicon
                    word = specials[word]    
                w = vocab.index(word)   # vocab integer to use as row number
                if w is not None:
                    matrix[w] = vector  # fill the vector into that row
                    seen.add(w)

    # Fill in OOL for any other vocab entries that were not seen in the lexicon.
    for w in range(len(vocab)):
        if w not in seen:
            matrix[w] = ool_vector

    log.info(f"From {file.name}, got embeddings for {len(seen)} of {len(vocab)} word types")

    return matrix

def log_counts_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a feature matrix with as many rows as corpus.vocab, where each
    row represents a feature vector for the corresponding word w.
    There is one feature (column) for each tag in corpus.tagset.  The value of this
    feature is log(1+c) where c=count(t,w) is the number of times t emitted w in supervised
    training data.  Thus, if this feature has weight 1.0 and is the only feature,
    then p(w | t) will be proportional to 1+count(t,w), just as in add-1 smoothing."""
    """Create a feature matrix where each row corresponds to a word in the corpus's vocabulary.
    Each column corresponds to a tag in the corpus's tagset.
    The value at a row and column is log(1 + count) where count is the number of times
    the tag (column) emitted the word (row) in the training data."""
    # Initialize emission counts
    emission_counts = {tag: Counter() for tag in corpus.tagset}

    # Calculate emission counts
    for sentence in corpus:
        for word, tag in sentence:
            if tag is not None:
                emission_counts[tag][word] += 1

    # Initialize an empty matrix with size vocab x tagset
    matrix = torch.zeros((len(corpus.vocab), len(corpus.tagset)))

    # Populate the matrix with log counts
    for tag_idx, tag in enumerate(corpus.tagset):
        for word_idx, word in enumerate(corpus.vocab):
            count = emission_counts[tag][word]
            matrix[word_idx, tag_idx] = np.log1p(count)

    return matrix

# def affixes_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
#     """Return a feature matrix with as many rows as corpus.vocab, where each
#     row represents a feature vector for the corresponding word w.
#     Each row has binary features for common suffixes and affixes that the
#     word has."""
#     """Create a feature matrix where each row corresponds to a word in the corpus's vocabulary.
#     Each column represents a binary feature for common prefixes and suffixes in that word."""
#     # List of common prefixes and suffixes
#     common_affixes = ['ing', 'ed', 's', 'es', 'ly', 'un', 're', 'pre', 'post', 'anti', 'de', 'trans', 'inter', 'non']

#     # Initialize an empty matrix with size vocab x number of affixes
#     matrix = torch.zeros((len(corpus.vocab), len(common_affixes)))

#     # Populate the matrix with binary values indicating affix presence
#     for word_idx, word in enumerate(corpus.vocab):
#         for affix_idx, affix in enumerate(common_affixes):
#             if word.startswith(affix) or word.endswith(affix):
#                 matrix[word_idx, affix_idx] = 1

#     return matrix

# def word_shape(word: str) -> str:
#     """
#     Generate a simplified word shape string from a word.
#     Converts uppercase letters to 'X', lowercase letters to 'x', digits to 'd', 
#     and other characters to their original form.
#     """
#     return re.sub(r'\d', 'd', re.sub(r'[A-Z]', 'X', re.sub(r'[a-z]', 'x', word)))

# def word_shape_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
#     """
#     Create a feature matrix where each row corresponds to a word in the corpus's vocabulary.
#     Each column represents a unique word shape found in the vocabulary.
#     Word shapes are simplified representations of the word structure.
#     """
#     # Generate word shapes for each word in the vocabulary
#     word_shapes = set(word_shape(word) for word in corpus.vocab)
#     word_shape_to_index = {shape: idx for idx, shape in enumerate(word_shapes)}

#     # Initialize an empty matrix with size vocab x number of unique word shapes
#     matrix = torch.zeros((len(corpus.vocab), len(word_shapes)))

#     # Populate the matrix with binary values indicating word shape presence
#     for word_idx, word in enumerate(corpus.vocab):
#         shape = word_shape(word)
#         shape_idx = word_shape_to_index[shape]
#         matrix[word_idx, shape_idx] = 1

#     return matrix
