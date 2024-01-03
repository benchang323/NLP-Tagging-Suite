#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Implementation of Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp, sqrt
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast

import torch
from torch import Tensor as Tensor
from torch import tensor as tensor
from torch import optim as optim
from torch import nn as nn
from torch import cuda as cuda
from torch.nn import functional as F
from jaxtyping import Float
from typeguard import typechecked
from tqdm import tqdm # type: ignore
import pickle
import logsumexp_safe

from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag,
                    TaggedCorpus, Word)
from integerize import Integerizer
from logsumexp_safe import logsumexp_new, logaddexp_new

TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class ConditionalRandomField(nn.Module):
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The unigram
        flag says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended
        to support higher-order HMMs: trigram HMMs used to be popular.)"""

        super().__init__() # type: ignore # pytorch nn.Module does not have type annotations

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)
        # As usual in Python, attributes starting with _ are intended as private;
        # in this case, they might go away if you changed the parametrization of the model.

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        assert vocab[-2:] == [EOS_WORD, BOS_WORD]  # make sure these are the last two

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.d = lexicon.size(1)   # dimensionality of a word's embedding in attribute space
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab
        self._E = lexicon  # embedding matrix; omits rows for EOS_WORD and BOS_WORD

        # Useful constants that are invoked in the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        assert self.bos_t is not None    # we need this to exist
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors
        
        self.init_params()     # create and initialize params


    @property
    def device(self) -> torch.device:
        """Get the GPU (or CPU) our tensors are computed and stored on."""
        return next(self.parameters()).device


    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> List[Tuple[int,Optional[int]]]:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        # Make sure that the sentence comes from a corpus that this HMM knows
        # how to handle.
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        # If so, go ahead and integerize it.
        return corpus.integerize_sentence(sentence)

    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).
        However, we initialize the BOS_TAG column of _WA to -inf, to ensure that
        we have 0 probability of transitioning to BOS_TAG (see "Don't guess when you know").
        See the "Parametrization" section of the reading handout."""

        # See the reading handout section "Parametrization."
        # 
        # As in HW3's probs.py, we wrap our model's parameters in nn.Parameter.
        # Then they will automatically be included in self.parameters(), which
        # this class inherits from nn.Module and which is used below for
        # regularization and training.

        ThetaB = 0.01*torch.rand(self.k, self.d)
        # ThetaB = ThetaB.log()
        self._ThetaB = nn.Parameter(ThetaB)    # params used to construct emission matrix

        WA = 0.01*torch.rand(1 if self.unigram # just one row if unigram model
                             else self.k,      # but one row per tag s if bigram model
                             self.k)           # one column per tag t
        WA[:, self.bos_t] = -inf               # correct the BOS_TAG column
        # WA = WA.log()
        self._WA = nn.Parameter(WA)            # params used to construct transition matrix


    @typechecked
    def params_L2(self) -> TorchScalar:
        """What's the L2 norm of the current parameter vector?
        We consider only the finite parameters."""
        l2 = tensor(0.0)
        for x in self.parameters():
            x_finite = x[x.isfinite()]
            l2 = l2 + x_finite @ x_finite   # add ||x_finite||^2
        return l2


    def updateAB(self) -> None:
        """Set the transition and emission matrices A and B, based on the current parameters.
        See the "Parametrization" section of the reading handout."""
        
        # A = F.softmax(self._WA, dim=1)       # run softmax on params to get transition distributions
        #                                      # note that the BOS_TAG column will be 0, but each row will sum to 1

        A = self._WA.exp()
        
        if self.unigram:
            # A is a row vector giving unigram probabilities p(t).
            # We'll just set the bigram matrix to use these as p(t | s)
            # for every row s.  This lets us simply use the bigram
            # code for unigram experiments, although unfortunately that
            # preserves the O(nk^2) runtime instead of letting us speed 
            # up to O(nk) in the unigram case.
            self.A = A.repeat(self.k, 1)
        else:
            # A is already a full matrix giving p(t | s).
            self.A = A

        WB = self._ThetaB @ self._E.t()  # inner products of tag weights and word embeddings
        # B = F.softmax(WB, dim=1)         # run softmax on those inner products to get emission distributions
        B = WB.exp()

        # WB = self._ThetaB @ self._E.t().log()
        # print(self._ThetaB @ self._E.t())
        # print(F.softmax(self._ThetaB @ self._E.t(), dim=1))
        # print(WB)
        # print(WB - logsumexp_new(WB, dim=1, safe_inf=True)[:, None])
        # B = WB - logsumexp_new(WB, dim=1, safe_inf=True)[:, None]
        self.B = B.clone()
        self.B[self.eos_t, :] = 1e-45        # but don't guess: EOS_TAG can't emit any column's word (only EOS_WORD)
        self.B[self.bos_t, :] = 1e-45        # same for BOS_TAG (although BOS_TAG will already be ruled out by other factors)

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")


    @typechecked
    def log_prob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        
        return self.log_forward(sentence, corpus) - self.log_forward(sentence.desupervise(), corpus)


    @typechecked
    def log_forward(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward 
        probability).

        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're 
        integerizing correctly."""

        sent = self._integerize_sentence(sentence, corpus)

        # The "nice" way to construct alpha is by appending to a List[Tensor] at each
        # step.  But to better match the notation in the handout, we'll instead preallocate
        # a list of length n+2 so that we can assign directly to alpha[j].


####################################################### Numerical
        
        alpha = [torch.zeros(self.k) for _ in sent]
        alpha[0][self.bos_t] = 1
        scales = torch.ones(len(sent)-1)
        
        # Most outer loop loops through all time steps; t<- 1 to n+1
        for j in range(1, len(sent)-1):  
            # alpha(j) is the one we are trying to fill in this loop
            word, tag = sent[j]  # observation and hidden state

            if tag is None: 
                support = torch.ones(self.k)
            else:
                support = torch.zeros(self.k)
                support[tag] = 1
                
            scales[j-1] = torch.max(alpha[j-1]).item()
            alpha[j] = alpha[j-1] * (1/ scales[j-1]) @ self.A * self.B[:, word] * support
            
        scales[-1] = torch.max(alpha[-2]).item()
        # Set alpha_EOS(n+1)=prob(obv_1, obv_2, ..., obv_n)
        alpha[-1][self.eos_t] = alpha[-2] * (1 / scales[-1]) @ self.A[:, self.eos_t]
        # print(alpha[n+1][self.eos_t].log() + torch.sum(scales.log()))

        return alpha[-1][self.eos_t].log() + torch.sum(scales.log())

########################################################## log domain
        
        # alpha = [torch.full((self.k,), float('-inf')) for _ in sent]
        # n = len(sent) - 2
        # alpha[0][self.bos_t] = 1e-45
        # # # scales = torch.zeros(n+1)
        # # print('numerical results:')
        # # print(f'alpha[0] = {alpha[0].exp()}')
        # # print(f'A = {self.A}')
        # # print(f'B[:, 0] = {self.B[:, 0]}')
        # # print(self.B)
        # # print(logsumexp_new(self.B, dim=1, safe_inf=True))
        # # print(self.B - logsumexp_new(self.B, dim=1, safe_inf=True)[:, None])
        # # print(alpha[0].exp() @ self.A)
        # # print(alpha[0].exp() @ self.A * self.B[:,0])
        # # print('log results:')
        # # print(f'alpha[0] = {alpha[0]}')
        # # print(f'A = {self.A.log()}')
        # # print(f'B[:, 0] = {self.B[:, 0].log()}')
        # # print(alpha[0][:, None] + self.A.log())
        # # print(torch.logsumexp(alpha[0][:, None] + self.A.log(), dim=0, keepdim=False, safe_inf=True))
        # # print(torch.logsumexp(alpha[0][:, None] + self.A.log(), dim=0, keepdim=False, safe_inf=True) + self.B[:, 0].log())

        
        # # Most outer loop loops through all time steps; t<- 1 to n+1
        # for j in range(1,n+1):  
        #     # alpha(j) is the one we are trying to fill in this loop
        #     obv, tag = sent[j]  # observation and hidden state
            
        #     # scales[j-1] = torch.max(alpha[j-1]).item()
        #         # print(alpha[j-1])
        #         # print(self.A + self.B[:, obv])
        #         # print(alpha[j-1] + self.A + self.B[:, obv])
        #     alpha[j] = logsumexp_new(alpha[j-1][:, None] + self.A, dim=3, safe_inf=True) + self.B[:, obv]

        #     # Efficient tag handling
        #     if tag is None: 
        #         sup = torch.zeros(self.k)
        #     else:
        #         support = torch.full((self.k,), float('-inf'))
        #         support[tag] = 1e-45
        #     alpha[j] = alpha[j] + support
            
        # #scales[n] = torch.max(alpha[n]).item()
        # # Set alpha_EOS(n+1)=prob(obv_1, obv_2, ..., obv_n)

        # # print(alpha[j-1] + self.A + self.B[:, self.eos_t])
        # alpha[n+1] = logsumexp_new(alpha[n][:, None] + self.A, dim=0, safe_inf=True)
        # # print(alpha[n+1][self.eos_t])

        # return alpha[n+1][self.eos_t]# + torch.sum(scales)

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # It continues to call the vector alpha, without the ^ that is added
        # in the handout.

        # We'll start by integerizing the input Sentence.
        # But make sure you deintegerize the words and tags again when
        # constructing the return value, since the type annotation on
        # this method says that it returns a Sentence object, and
        # that's what downstream methods like eval_tagging will
        # expect.  (Running mypy on your code will check that your
        # code conforms to the type annotations ...)
        sent = self._integerize_sentence(sentence, corpus)

        alpha = [torch.empty(self.k) for _ in sent]
        alpha[0][self.bos_t] = 1  # Start of sentence
        backpoints = []
        scales = torch.ones(len(sent)-1)  # Initialize scales
        
        # Iterating through each word in the sentence
        for j in range(1, len(sent) - 1):
            word, tag = sent[j]
            max_value, max_index = torch.max(((alpha[j-1].repeat(self.k, 1)).t() * self.A), dim=0)

            if tag is None: 
                support = torch.ones(self.k)
            else:
                support = torch.zeros(self.k)
                support[tag] = 1
                
            scales[j-1] = torch.max(alpha[j-1]).item()
            max_value = max_value * self.B[:, word] * (1 / scales[j-1]) * support
            # max_value = max_value * self.B[:, word] * support
            
            alpha[j] = max_value
            backpoints.append(max_index)

        backpoint_tag = torch.argmax(alpha[-2])
        tag_list = [backpoint_tag]
        for backpoint_list in reversed(backpoints):
            backpoint_tag = backpoint_list[backpoint_tag]
            tag_list.append(backpoint_tag)
        tag_list.reverse()
        tag_list.append(self.eos_t) # Add end of sentence (EOS) tag.

        # Building the result sentence using the Sentence class.
        result_sentence = Sentence()
        # Add start of sentence (BOS) tag.
        result_sentence.append((sentence[0][0], corpus.tagset[self.bos_t]))

        # Skip the first and last tags in tag_list as they are BOS and EOS
        for i, (word, _) in enumerate(sentence[1:-1], start=1):
            # Deintegerize the tag
            tag = corpus.tagset[tag_list[i]]
            # Append the word and its most probable tag to the result sentence
            result_sentence.append((word, tag))

        # Finally, add the end of sentence (EOS) tag
        result_sentence.append((sentence[-1][0], corpus.tagset[self.eos_t]))

        return result_sentence


            
    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[CRF], float],
              tolerance: float =0.001,
              minibatch_size: int = 1,
              evalbatch_size: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              save_path: Path = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        The minibatch size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus.)
        The evalbatch size controls how often we evaluate (e.g., on a development corpus).
        We will stop when the relative improvement of the evaluation loss,
        since the last evalbatch, is less than the tolerance; in particular,
        we will stop when the improvement is negative, i.e., the evaluation loss 
        is getting worse (overfitting).
        lr is the learning rate, and reg is an L2 batch regularization coefficient."""

        # This is relatively generic training code.  Notice however that the
        # updateAB step before each minibatch produces A, B matrices that are
        # then shared by all sentences in the minibatch.

        # All of the sentences in a minibatch could be treated in parallel,
        # since they use the same parameters.  The code below treats them
        # in series, but if you were using a GPU, you could get speedups
        # by writing the forward algorithm using higher-dimensional tensor 
        # operations that update alpha[j-1] to alpha[j] for all the sentences
        # in the minibatch at once, and then PyTorch could actually take
        # better advantage of hardware parallelism.

        assert minibatch_size > 0
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        assert reg >= 0

        old_dev_loss: Optional[float] = None    # we'll keep track of the dev loss here

        optimizer = optim.SGD(self.parameters(), lr=lr)  # optimizer knows what the params are
        self.updateAB()                                        # compute A and B matrices from current params
        log_likelihood = tensor(0.0, device=self.device)       # accumulator for minibatch log_likelihood
        for m, sentence in tqdm(enumerate(corpus.draw_sentences_forever())):
            # Before we process the new sentence, we'll take stock of the preceding
            # examples.  (It would feel more natural to do this at the end of each
            # iteration instead of the start of the next one.  However, we'd also like
            # to do it at the start of the first time through the loop, to print out
            # the dev loss on the initial parameters before the first example.)

            # m is the number of examples we've seen so far.
            # If we're at the end of a minibatch, do an update.
            if m % minibatch_size == 0 and m > 0:
                logger.debug(f"Training log-likelihood per example: {log_likelihood.item()/minibatch_size:.3f} nats")
                optimizer.zero_grad()          # backward pass will add to existing gradient, so zero it
                objective = -log_likelihood + (minibatch_size/corpus.num_tokens()) * reg * self.params_L2()
                objective.backward()           # type: ignore # compute gradient of regularized negative log-likelihod
                length = sqrt(sum((x.grad*x.grad).sum().item() for x in self.parameters()))
                logger.debug(f"Size of gradient vector: {length}")  # should approach 0 for large minibatch at local min
                optimizer.step()               # SGD step
                self.updateAB()                # update A and B matrices from new params
                log_likelihood = tensor(0.0, device=self.device)    # reset accumulator for next minibatch

            # If we're at the end of an eval batch, or at the start of training, evaluate.
            if m % evalbatch_size == 0:
                with torch.no_grad():       # type: ignore # don't retain gradients during evaluation
                    dev_loss = loss(self)   # this will print its own log messages
                if old_dev_loss is not None and dev_loss >= old_dev_loss * (1-tolerance):
                    # we haven't gotten much better, so stop
                    self.save(save_path)  # Store this model, in case we'd like to restore it later.
                    break
                old_dev_loss = dev_loss            # remember for next eval batch

            # Finally, add likelihood of sentence m to the minibatch objective.
            log_likelihood = log_likelihood + self.log_prob(sentence, corpus)


    def save(self, model_path: Path) -> None:
        logger.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path}")


    @classmethod
    def load(cls, model_path: Path, gpu: bool = False) -> CRF:
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=('cuda' if gpu else 'cpu'))
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from saved file {model_path}.")
        logger.info(f"Loaded model from {model_path}")
        return model



class ConditionalRandomFieldBiRNN(ConditionalRandomField):
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The unigram
        flag says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended
        to support higher-order HMMs: trigram HMMs used to be popular.)"""
        self.hidden_size = 128
        torch.autograd.set_detect_anomaly(True)
        super().__init__(tagset,vocab,lexicon) # type: ignore # pytorch nn.Module does not have type annotations

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)
        # As usual in Python, attributes starting with _ are intended as private;
        # in this case, they might go away if you changed the parametrization of the model.

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        # assert vocab[-2:] == [EOS_WORD, BOS_WORD]  # make sure these are the last two

        # self.k = len(tagset)       # number of tag types
        # self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        # self.d = lexicon.size(1)   # dimensionality of a word's embedding in attribute space
        # self.unigram = unigram     # do we fall back to a unigram model?
        # self.hidden_size = 128
        # self.tagset = tagset
        # self.vocab = vocab
        # self._E = lexicon[:-2]  # embedding matrix; omits rows for EOS_WORD and BOS_WORD

        # # Useful constants that are invoked in the methods
        # self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        # self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        # assert self.bos_t is not None    # we need this to exist
        # assert self.eos_t is not None    # we need this to exist
        # self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors
        
        # self.init_params()     # create and initialize params


    @property
    def device(self) -> torch.device:
        """Get the GPU (or CPU) our tensors are computed and stored on."""
        return next(self.parameters()).device


    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> List[Tuple[int,Optional[int]]]:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        # Make sure that the sentence comes from a corpus that this HMM knows
        # how to handle.
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        # If so, go ahead and integerize it.
        return corpus.integerize_sentence(sentence)

    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).
        However, we initialize the BOS_TAG column of _WA to -inf, to ensure that
        we have 0 probability of transitioning to BOS_TAG (see "Don't guess when you know").
        See the "Parametrization" section of the reading handout."""

        # See the reading handout section "Parametrization."
        # 
        # As in HW3's probs.py, we wrap our model's parameters in nn.Parameter.
        # Then they will automatically be included in self.parameters(), which
        # this class inherits from nn.Module and which is used below for
        # regularization and training.

        ThetaB = 0.01*torch.rand(self.k, self.hidden_size)
        # ThetaB = ThetaB.log()
        self._ThetaB = nn.Parameter(ThetaB)    # params used to construct emission matrix

        WA = 0.01*torch.rand(1 if self.unigram # just one row if unigram model
                             else self.k,      # but one row per tag s if bigram model
                             self.hidden_size)           # one column per tag t
        WA[:, self.bos_t] = 1e-45               # correct the BOS_TAG column
        # WA = WA.log()
        self._WA = nn.Parameter(WA)            # params used to construct transition matrix

        # M is used to calculate hidden layers
        self.M = nn.Linear(in_features=self.hidden_size+self.d, out_features=self.hidden_size)
        self.M_b = nn.Linear(in_features=self.hidden_size+self.d, out_features=self.hidden_size)
        # get tag embeddings
        self.embed_t = nn.Embedding(self.k,self.hidden_size)
        # U has bias parameters, 2 hidden layers, 
        self.U_a = nn.Linear(in_features=4*self.hidden_size, out_features=self.hidden_size)
        self.U_b = nn.Linear(in_features=3*self.hidden_size + self.d, out_features=self.hidden_size)


    @typechecked
    def params_L2(self) -> TorchScalar:
        """What's the L2 norm of the current parameter vector?
        We consider only the finite parameters."""
        l2 = tensor(0.0)
        for x in self.parameters():
            x_finite = x[x.isfinite()]
            l2 = l2 + x_finite @ x_finite   # add ||x_finite||^2
        return l2


    def updateAB(self) -> None:
        """Set the transition and emission matrices A and B, based on the current parameters.
        See the "Parametrization" section of the reading handout."""
        
        # A = F.softmax(self._WA, dim=1)       # run softmax on params to get transition distributions
        #                                      # note that the BOS_TAG column will be 0, but each row will sum to 1

        A = self._WA
        
        if self.unigram:
            # A is a row vector giving unigram probabilities p(t).
            # We'll just set the bigram matrix to use these as p(t | s)
            # for every row s.  This lets us simply use the bigram
            # code for unigram experiments, although unfortunately that
            # preserves the O(nk^2) runtime instead of letting us speed 
            # up to O(nk) in the unigram case.
            self.A = A.repeat(self.k, 1)
        else:
            # A is already a full matrix giving p(t | s).
            self.A = A.clone()

        WB = self._ThetaB  # inner products of tag weights and word embeddings
        # B = F.softmax(WB, dim=1)         # run softmax on those inner products to get emission distributions
        B = WB

        # WB = self._ThetaB @ self._E.t().log()
        # print(self._ThetaB @ self._E.t())
        # print(F.softmax(self._ThetaB @ self._E.t(), dim=1))
        # print(WB)
        # print(WB - logsumexp_new(WB, dim=1, safe_inf=True)[:, None])
        # B = WB - logsumexp_new(WB, dim=1, safe_inf=True)[:, None]
        self.B = B.clone()
        self.B[self.eos_t, :] = 1e-45        # but don't guess: EOS_TAG can't emit any column's word (only EOS_WORD)
        self.B[self.bos_t, :] = 1e-45        # same for BOS_TAG (although BOS_TAG will already be ruled out by other factors)

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")


    @typechecked
    def log_prob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        log_prob = -1*self.log_forward(sentence, corpus)
        return log_prob

    @typechecked
    def log_forward(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward 
        probability).

        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're 
        integerizing correctly."""

        sent = self._integerize_sentence(sentence, corpus)

        # The "nice" way to construct alpha is by appending to a List[Tensor] at each
        # step.  But to better match the notation in the handout, we'll instead preallocate
        # a list of length n+2 so that we can assign directly to alpha[j].


####################################################### Numerical
        
        alpha = [torch.zeros(self.k) for _ in sent]
        alpha[0][self.bos_t] = 1
        scales = [1]*(len(sent)-1)
        h_curr = torch.ones(self.hidden_size, requires_grad=True) * 1e-45
        h_b_curr = torch.ones(self.hidden_size, requires_grad=True) * 1e-45
        h_f = [h_curr]
        h_back = [h_b_curr]
        # initialize hidden weights
        for j in range(len(sent)):
            s_word, _ = sent[j]
            e_word, _ = sent[-(j+1)]

            h_curr = F.sigmoid(self.M(torch.concat((h_curr, self._E[s_word]), dim=0)))
            h_f.append(h_curr)
            h_b_curr = F.sigmoid(self.M_b(torch.concat((h_b_curr, self._E[e_word]), dim=0)))
            h_back.append(h_b_curr)
        # Most outer loop loops through all time steps; t<- 1 to n+1
        for j in range(1, len(sent)):  
            # alpha(j) is the one we are trying to fill in this loop
            word, tag = sent[j]  # observation and hidden state
            _, s = sent[j-1]
            s_embed = self.embed_t(torch.tensor(s))
            t_embed = self.embed_t(torch.tensor(tag))
            h_back_curr = h_back[-(j+1)]

            f_a = F.sigmoid(self.U_a(torch.concat((h_f[j-1], s_embed, t_embed, h_back_curr)))).reshape(1,-1)
            f_b = F.sigmoid(self.U_b(torch.concat((h_f[j], t_embed, self._E[word], h_back_curr)))).reshape(1,-1)
            # if tag is None: 
            #     support = torch.ones(self.k)
            # else:
            #     support = torch.zeros(self.k)
            #     support[tag] = 1
                

            scales[j-1] = torch.max(alpha[j-1]).item()
            initial_b =  (self.B * f_b).exp().t()
            initial = (self.A * f_a).exp()

            combined = initial @ initial_b
            alpha[j] = ((alpha[j-1] / scales[j-1])*combined).sum(dim=1)
            # alpha[j] = ((alpha[j-1] / scales[j-1]) * ((self.A * f_a).exp() * (self.B * f_b).exp()))#.sum(dim=0)
            # print(alpha[j])

        return alpha[-1][self.eos_t].log() + torch.sum(torch.tensor(scales).log())

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # It continues to call the vector alpha, without the ^ that is added
        # in the handout.

        # We'll start by integerizing the input Sentence.
        # But make sure you deintegerize the words and tags again when
        # constructing the return value, since the type annotation on
        # this method says that it returns a Sentence object, and
        # that's what downstream methods like eval_tagging will
        # expect.  (Running mypy on your code will check that your
        # code conforms to the type annotations ...)
        sent = self._integerize_sentence(sentence, corpus)


        backpoints = []
        alpha = [torch.zeros(self.k) for _ in sent]
        alpha[0][self.bos_t] = 1
        scales = [1]*(len(sent)-1)
        h_curr = torch.ones(self.hidden_size, requires_grad=True) * 1e-45
        h_b_curr = torch.ones(self.hidden_size, requires_grad=True) * 1e-45
        h_f = [h_curr]
        h_back = [h_b_curr]
        # initialize hidden weights
        for j in range(len(sent)):
            s_word, _ = sent[j]
            e_word, _ = sent[-(j+1)]

            h_curr = F.sigmoid(self.M(torch.concat((h_curr, self._E[s_word]), dim=0)))
            h_f.append(h_curr)
            h_b_curr = F.sigmoid(self.M_b(torch.concat((h_b_curr, self._E[e_word]), dim=0)))
            h_back.append(h_b_curr)
        # Most outer loop loops through all time steps; t<- 1 to n+1
        for j in range(1, len(sent)):  
            # alpha(j) is the one we are trying to fill in this loop
            word, tag = sent[j]  # observation and hidden state
            _, s = sent[j-1]
            s_embed = self.embed_t(torch.tensor(s))
            t_embed = self.embed_t(torch.tensor(tag))
            h_back_curr = h_back[-(j+1)]

            f_a = F.sigmoid(self.U_a(torch.concat((h_f[j-1], s_embed, t_embed, h_back_curr)))).reshape(1,-1)
            f_b = F.sigmoid(self.U_b(torch.concat((h_f[j], t_embed, self._E[word], h_back_curr)))).reshape(1,-1)
            # if tag is None: 
            #     support = torch.ones(self.k)
            # else:
            #     support = torch.zeros(self.k)
            #     support[tag] = 1
                

            scales[j-1] = torch.max(alpha[j-1]).item()
            initial_b =  (self.B * f_b).exp().t()
            initial = (self.A * f_a).exp()

            combined = initial @ initial_b
            alpha[j], max_index = ((alpha[j-1] / scales[j-1])*combined).sum(dim=1)
            backpoints.append[max_index]


        backpoint_tag = torch.argmax(alpha[-2])
        tag_list = [backpoint_tag]
        for backpoint_list in reversed(backpoints):
            backpoint_tag = backpoint_list[backpoint_tag]
            tag_list.append(backpoint_tag)
        tag_list.reverse()
        tag_list.append(self.eos_t) # Add end of sentence (EOS) tag.

        # Building the result sentence using the Sentence class.
        result_sentence = Sentence()
        # Add start of sentence (BOS) tag.
        result_sentence.append((sentence[0][0], corpus.tagset[self.bos_t]))

        # Skip the first and last tags in tag_list as they are BOS and EOS
        for i, (word, _) in enumerate(sentence[1:-1], start=1):
            # Deintegerize the tag
            tag = corpus.tagset[tag_list[i]]
            # Append the word and its most probable tag to the result sentence
            result_sentence.append((word, tag))

        # Finally, add the end of sentence (EOS) tag
        result_sentence.append((sentence[-1][0], corpus.tagset[self.eos_t]))

        return result_sentence


            
    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[CRF], float],
              tolerance: float =0.001,
              minibatch_size: int = 1,
              evalbatch_size: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              save_path: Path = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        The minibatch size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus.)
        The evalbatch size controls how often we evaluate (e.g., on a development corpus).
        We will stop when the relative improvement of the evaluation loss,
        since the last evalbatch, is less than the tolerance; in particular,
        we will stop when the improvement is negative, i.e., the evaluation loss 
        is getting worse (overfitting).
        lr is the learning rate, and reg is an L2 batch regularization coefficient."""

        # This is relatively generic training code.  Notice however that the
        # updateAB step before each minibatch produces A, B matrices that are
        # then shared by all sentences in the minibatch.

        # All of the sentences in a minibatch could be treated in parallel,
        # since they use the same parameters.  The code below treats them
        # in series, but if you were using a GPU, you could get speedups
        # by writing the forward algorithm using higher-dimensional tensor 
        # operations that update alpha[j-1] to alpha[j] for all the sentences
        # in the minibatch at once, and then PyTorch could actually take
        # better advantage of hardware parallelism.

        assert minibatch_size > 0
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        assert reg >= 0

        old_dev_loss: Optional[float] = None    # we'll keep track of the dev loss here

        optimizer = optim.SGD(self.parameters(), lr=lr)  # optimizer knows what the params are
        self.updateAB()                                        # compute A and B matrices from current params
        log_likelihood = tensor(0.0, device=self.device)       # accumulator for minibatch log_likelihood
        for m, sentence in tqdm(enumerate(corpus.draw_sentences_forever())):
            # Before we process the new sentence, we'll take stock of the preceding
            # examples.  (It would feel more natural to do this at the end of each
            # iteration instead of the start of the next one.  However, we'd also like
            # to do it at the start of the first time through the loop, to print out
            # the dev loss on the initial parameters before the first example.)
            # print([x for x in self.parameters()])
            # m is the number of examples we've seen so far.
            # If we're at the end of a minibatch, do an update.
            if m % minibatch_size == 0 and m > 0:
                logger.debug(f"Training log-likelihood per example: {log_likelihood.item()/minibatch_size:.3f} nats")
                optimizer.zero_grad()          # backward pass will add to existing gradient, so zero it
                objective = -log_likelihood + (minibatch_size/corpus.num_tokens()) * reg * self.params_L2()
                objective.backward()           # type: ignore # compute gradient of regularized negative log-likelihod
                length = sqrt(sum((x.grad*x.grad).sum().item() for x in self.parameters()))
                logger.debug(f"Size of gradient vector: {length}")  # should approach 0 for large minibatch at local min
                optimizer.step()               # SGD step
                self.updateAB()                # update A and B matrices from new params
                log_likelihood = tensor(0.0, device=self.device)    # reset accumulator for next minibatch

            # If we're at the end of an eval batch, or at the start of training, evaluate.
            if m % evalbatch_size == 0:
                with torch.no_grad():       # type: ignore # don't retain gradients during evaluation
                    dev_loss = loss(self)   # this will print its own log messages
                if old_dev_loss is not None and dev_loss >= old_dev_loss * (1-tolerance):
                    # we haven't gotten much better, so stop
                    self.save(save_path)  # Store this model, in case we'd like to restore it later.
                    break
                old_dev_loss = dev_loss            # remember for next eval batch

            # Finally, add likelihood of sentence m to the minibatch objective.
            log_likelihood = log_likelihood + self.log_prob(sentence, corpus)


    def save(self, model_path: Path) -> None:
        logger.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path}")


    @classmethod
    def load(cls, model_path: Path, gpu: bool = False) -> CRF:
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=('cuda' if gpu else 'cpu'))
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from saved file {model_path}.")
        logger.info(f"Loaded model from {model_path}")
        return model