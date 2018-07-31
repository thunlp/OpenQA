#!/usr/bin/env python3

"""OpenQA model"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy
import random

import torch.nn as nn

from torch.autograd import Variable
from .config import override_model_args
from .rnn_reader import RnnDocReader
from .rnn_selector import RnnDocSelector

from src.reader import vector

type_max = True;

logger = logging.getLogger(__name__)


class DocReader(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, word_dict, feature_dict,
                 state_dict=None,  normalize=False, state_dict_selector = None):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        logger.info("vocab_size:\t%d", self.args.vocab_size)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)
        self.selector = RnnDocSelector(args)
        self.selector.embedding.weight.data.copy_(self.network.embedding.weight.data)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)
        if state_dict_selector:
            self.selector.load_state_dict(state_dict_selector)

    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        logger.info("init_optimizer")
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
            for p in self.selector.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad] 
        parameters = parameters + [p for p in self.selector.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------
    def get_score(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.eval()

        batch_size = ex[0].size(0)
        # Transfer to GPU
        inputs = [e if e is None else Variable(e.cuda(async=True))
                  for e in ex[:5]]

        # Run forward
        score_s, score_e, _, _ = self.network(*inputs)
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        return score_s, score_e
    def update(self, ex, target_s, target_e, HasAnswer_list):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        batch_size = ex[0].size(0)
        # Transfer to GPU
        inputs = [e if e is None else Variable(e.cuda(async=True))
                  for e in ex[:5]]

        # Run forward
        score_s, score_e, _, _ = self.network(*inputs)

        # Compute loss and accuracies
        loss = Variable(torch.FloatTensor([0.0]).cuda())
        flag = False
        num_items1, num_items2 = 0, 0
        loss1, loss2 = 0, 0
        for i in range(0, batch_size):
            if HasAnswer_list[i][0]: 
                flag = True
                try:
                    tmp1 = score_s[i][target_s[i][0][0]]*score_e[i][target_s[i][0][1]]
                    for j in range(1, len(target_s[i])):
                        if (type_max):
                            if (tmp1.data.cpu().numpy()[0]<(score_s[i][target_s[i][j][0]]*score_e[i][target_s[i][j][1]]).data.cpu().numpy()[0]):
                                tmp1 = score_s[i][target_s[i][j][0]]*score_e[i][target_s[i][j][1]]
                        else:
                            tmp1 += score_s[i][target_s[i][j][0]]*score_e[i][target_s[i][j][1]]
                    loss1 = loss1 - (tmp1+1e-16).log()
                    num_items1 +=1
                except:
                    logger.info(score_s[i].size(0))
                    logger.info(score_e[i].size(0))
                    logger.info(ex[0][i].size(0))
                    logger.info(target_s[i])
                    logger.info(target_e[i])
            else:
                num_items2 +=1
                tmp2 = 0
                for j in range(len(target_s[i])):
                    tmp2 = tmp2 + score_s[i][target_s[i][j][0]]*score_e[i][target_s[i][j][1]]
                loss2 = loss2 - (1-tmp2+1e-16).log()
            
        # Clear gradients and run backward
        if num_items1>0:
            loss += loss1/num_items1

        #if num_items2>0:
        #    loss += 0.5*loss2/num_items2
        self.optimizer.zero_grad()
        if flag:
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                          self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], ex[0].size(0)

    def update_with_doc(self, update_step, ex_with_doc, pred_s_list_doc, pred_e_list_doc, top_n, target_s_list, target_e_list, HasAnswer_list):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.network.train()
        self.selector.train()
        batch_size = ex_with_doc[0][0].size(0)
        num_docs = int(vector.num_docs/3)
        for idx_doc in range(num_docs):
            pred_s_list_doc[idx_doc] = pred_s_list_doc[idx_doc].cuda(async=True)
            pred_e_list_doc[idx_doc] = pred_e_list_doc[idx_doc].cuda(async=True)
        scores_doc = Variable(torch.zeros(batch_size, vector.num_docs).cuda())
        scores_doc_norm = Variable(torch.zeros(batch_size, vector.num_docs).cuda())
        inputs_list = []
        for idx_doc in range(num_docs):
        # Transfer to GPU
            ex = ex_with_doc[idx_doc]
            inputs = [e if e is None else
                      Variable(e.cuda(async=True))
                      for e in ex[:5]]
            inputs_list.append(inputs)
            scores_doc[:, idx_doc] = self.selector(*inputs)
        for i in range(batch_size):
            scores_doc_norm[i] = F.softmax(scores_doc[i])
        loss = Variable(torch.FloatTensor([0.0]).cuda())
        loss_by_batch = [0.0 for i in range(batch_size)]
        loss1 = Variable(torch.FloatTensor([0.0]).cuda())
        num1 = 0
        flag = [False for i in range(batch_size)]
        tot_flag = False
        num_answer  = [0.0 for i in range(batch_size)]
        for i in range(batch_size):
            for idx_doc in range(num_docs):
                num_answer[i] += HasAnswer_list[idx_doc][i][0]
        for idx_doc in range(num_docs):    
            # Run forward
            inputs = inputs_list[idx_doc]
            score_s, score_e,_,_ = self.network(*inputs)
            for i in range(batch_size):
                if (HasAnswer_list[idx_doc][i][0]):
                    loss += 0.5*Variable(torch.FloatTensor([1.0/num_answer[i]]).cuda()) * (-(scores_doc_norm[i][idx_doc]+1e-16).log() + Variable(torch.FloatTensor([1.0/num_answer[i]]).cuda().log()))
                    tmp1 = score_s[i][target_s_list[idx_doc][i][0][0]]*score_e[i][target_s_list[idx_doc][i][0][1]]
                    for j in range(1, len(target_s_list[idx_doc][i])):
                        if (type_max):
                            if (tmp1.data.cpu().numpy()[0]<( score_s[i][target_s_list[idx_doc][i][j][0]]*score_e[i][target_s_list[idx_doc][i][j][1]]).data.cpu().numpy()[0]):
                                tmp1 =  score_s[i][target_s_list[idx_doc][i][j][0]]*score_e[i][target_s_list[idx_doc][i][j][1]]
                        else:
                            tmp1 +=  score_s[i][target_s_list[idx_doc][i][j][0]]*score_e[i][target_s_list[idx_doc][i][j][1]]
                    loss_by_batch[i] += tmp1*scores_doc_norm[i][idx_doc]
                    flag[i] = True
                    
        num_items1 = 0
        for i in range(batch_size):
            if (flag[i]):
                num_items1+=1
        for i in range(batch_size):
            if (flag[i]):
                loss-=1.0/num_items1*((loss_by_batch[i]+1e-16).log())
                tot_flag = True
        if (num1>0):
            tot_flag = True
            loss-=1.0/num1*loss1
        self.optimizer.zero_grad()
        if tot_flag:
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                          self.args.grad_clipping)
            # Update parameters
            self.optimizer.step()
            self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], batch_size
    def pretrain_selector(self, ex_with_doc, HasAnswer_list):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.network.train()
        self.selector.train()
        batch_size = ex_with_doc[0][0].size(0)
        scores_doc = Variable(torch.zeros(batch_size, vector.num_docs).cuda())
        scores_doc_norm = Variable(torch.zeros(batch_size, vector.num_docs).cuda())
        scores_doc_norm1 = Variable(torch.zeros(batch_size, vector.num_docs).cuda())
        num_docs = int(vector.num_docs)
        for idx_doc in range(num_docs):
        # Transfer to GPU
            ex = ex_with_doc[idx_doc]
            inputs = [e if e is None else
                      Variable(e.cuda(async=True))
                      for e in ex[:5]]
            scores_doc[:, idx_doc] = self.selector(*inputs)#.pow(2)
        for i in range(batch_size):
            scores_doc_norm[i] = F.softmax(scores_doc[i])
        loss = Variable(torch.FloatTensor([0.0]).cuda())
        loss1 = 0
        loss2 = 0
        num_items1 = 0
        num_items2 = 0
        flag = False
        for i in range(batch_size):
            s = ''
            num_answer = 0
            for idx_doc in range(num_docs):
                num_answer += HasAnswer_list[idx_doc][i]
            flag1 = False
            tmp1 = 0
            for idx_doc in range(num_docs):
                if (HasAnswer_list[idx_doc][i]==1):
                    flag = True
                    if (scores_doc_norm[i][idx_doc].data.cpu().numpy()[0]>1e-16):
                        loss += Variable(torch.FloatTensor([1.0/num_answer]).cuda()) * (-(scores_doc_norm[i][idx_doc]+1e-16).log() + Variable(torch.FloatTensor([1.0/num_answer]).cuda().log()))
                   
                    tmp1 += scores_doc_norm[i][idx_doc]
            if (flag1):
                loss1 -= (tmp1+1e-16).log()
                num_items1+=1
        if (num_items1>0):
            loss+= loss1/num_items1
        self.optimizer.zero_grad()
        if flag:
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                          self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], batch_size

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            # Embeddings to fix are indexed after the special + N tuned words
            offset = self.args.tune_partial + self.word_dict.START
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding
            if offset < embedding.size(0):
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------
    def predict_with_doc(self, ex_with_doc):
        self.selector.eval()
        self.network.eval()
        batch_size = ex_with_doc[0][0].size(0)

        scores_doc = Variable(torch.zeros(batch_size, vector.num_docs).cuda())
        scores_doc_norm = Variable(torch.zeros(batch_size, vector.num_docs).cuda())
        for idx_doc in range(vector.num_docs):
            ex = ex_with_doc[idx_doc]
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
            scores_doc[:, idx_doc] = self.selector(*inputs)
        for i in range(batch_size):
            scores_doc_norm[i] = F.softmax(scores_doc[i])

        return scores_doc_norm.data.cpu() 
    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        """Forward a batch of examples only to get predictions.

        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex[:5]]

        # Run forward
        score_s, score_e, _, _ = self.network(*inputs)

        # Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_s, score_e, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.

        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e. Except only consider
        spans that are in the candidates list.
        """
        pred_s = []
        pred_e = []
        pred_score = []
        for i in range(score_s.size(0)):
            # Extract original tokens stored with candidates
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            if not cands:
                # try getting from globals? (multiprocessing in pipeline mode)
                from ..pipeline.drqa import PROCESS_CANDS
                cands = PROCESS_CANDS
            if not cands:
                raise RuntimeError('No candidates given.')

            # Score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_strings=False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    # Match! Record its score.
                    scores.append(score_s[i][s] * score_e[i][e - 1])
                    s_idx.append(s)
                    e_idx.append(e - 1)

            if len(scores) == 0:
                # No candidates present
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                # Rank found candidates
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
        return pred_s, pred_e, pred_score

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        state_dict_selector = copy.copy(self.selector.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'state_dict_selector': state_dict_selector,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        try:
            state_dict_selector = saved_params['state_dict_selector']
            logger.info("load_pretrained_selector")
            return DocReader(args, word_dict, feature_dict, state_dict, normalize, state_dict_selector)
        except:
            return DocReader(args, word_dict, feature_dict, state_dict, normalize)

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = DocReader(args, word_dict, feature_dict, state_dict, normalize)
        model.init_optimizer(optimizer)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()
        self.selector = self.selector.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()
        self.selector = self.selector.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
