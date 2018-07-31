#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Edit from DrQA"""

import numpy as np
import logging
import unicodedata

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .vector import vectorize, vectorize_with_doc
from .vector import num_docs

import json

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class ReaderDataset(Dataset):

    def __init__(self, examples, model, single_answer=False):
        self.model = model
        self.examples = examples
        self.single_answer = single_answer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        
        return vectorize(self.examples[index], self.model, self.single_answer)

    def lengths(self):
        return [(len(ex['document']), len(ex['question']))
                for ex in self.examples]


def has_answer(answer, text):
    """Check if a text contains an answer string."""

    for single_answer in answer:
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True
    return False

class ReaderDataset_with_Doc(Dataset):

    def __init__(self, examples, model, docs, single_answer=False):
        self.model = model
        self.examples = examples
        self.single_answer = single_answer
        self.docs = docs
        #for i in range(len(self.examples)):
        #    for j in range(0, len(self.docs_by_question[i])):
        #        self.docs_by_question[i]['has_answer'] = has_answer(self.examples[i]['answer'], self.docs_by_question[i][document])
        #print (self.docs_by_question.keys())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        question = self.examples[index]['question']
        #logger.info("%d\t%s",index, question)
        #logger.info(self.docs_by_question[question])
        #assert("\n" not in question)
        #if (question not in self.docs_by_question):
        #    logger.info("No find question:%s", question)
        #    return []
        return vectorize_with_doc(self.examples[index], index, self.model, self.single_answer, self.docs[index])

    def lengths(self):
        #return [(len(ex['document']), len(ex['question']))
        #        for ex in self.examples]
        return [(len(doc[num_docs-1]['document']), len(doc[num_docs-1]['question'])) for doc in self.docs]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
