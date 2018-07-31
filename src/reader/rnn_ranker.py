#!/usr/bin/env python3


"""Implementation of the Paragraph Selector."""

import torch
import torch.nn as nn
from . import layers

import torch.nn.functional as F


import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocRanker(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(RnnDocRanker, self).__init__()
        # Store config
        self.args = args
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        
        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        #self.args.dropout_emb = 0
        my_layer_num = 1
        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=my_layer_num,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=nn.LSTM,
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=my_layer_num,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=nn.LSTM,
            padding=args.rnn_padding,
        )
        

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size

        if args.concat_rnn_layers:
            doc_hidden_size *= my_layer_num
            question_hidden_size *= my_layer_num

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end

        self.ans_attn = layers.BilinearSeqAttn1(
            doc_hidden_size,
            question_hidden_size,
        )
        self.ans_attn1 = layers.BilinearSeqAttn1(
            doc_hidden_size,
            question_hidden_size,
        )
        self.ans_attn2 = layers.BilinearSeqAttn2(
        )
        self.dense1 = nn.Linear(args.embedding_dim, doc_hidden_size)
        self.dense2 = nn.Linear(args.embedding_dim, question_hidden_size)

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)
        '''
        doc_hiddens = F.tanh(self.dense1(x1_emb))
        question_hiddens = F.tanh(self.dense2(x2_emb))
        '''
        # Encode document with RNN
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask) # batch * len1 * him

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask) # batch * len2 * him

        '''
        dq_mat_column, dq_mat_row= self.ans_attn2(doc_hiddens,question_hiddens, x1_mask, x2_mask) # batch * len1 * len2
        #logger.info(dq_mat_column)
        #logger.info(dq_mat_row)
        dq_mat_row_norm = layers.weighted_avg(dq_mat_row, layers.uniform_weights(dq_mat_row, x1_mask)) # weights: batch * len1 \ avg: batch*len2
        #logger.info(dq_mat_row_norm)
        output = dq_mat_column.bmm(dq_mat_row_norm.unsqueeze(2)).squeeze(2) #batch*len1
        #logger.info(output)
        scores = torch.max(output, 1)[0]
        #logger.info(scores)
        '''
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        # Predict start and end positions
        scores = torch.max(self.ans_attn(doc_hiddens, question_hidden, x1_mask), 1)[0]#.sigmoid()
        scores = scores + torch.max(self.ans_attn1(doc_hiddens, question_hidden, x1_mask), 1)[0]

        #output_hidden = self.ans_attn(doc_hiddens, question_hidden, x1_mask).view(x1_mask.size(0), -1, 1)
        #scores = layers.weighted_avg(output_hidden, layers.uniform_weights(output_hidden, x1_mask))
        #logger.info(layers.uniform_weights(output_hidden, x1_mask))
        #logger.info(output_hidden)
        #logger.info(scores)
        '''
        dq = self.ans_attn2(doc_hiddens,question_hiddens, x1_mask, x2_mask).view(x1_mask.size(0), -1)
        sorted, indices = dq.sort(1,descending = True)
        scores = sorted[:,:1].sum(1)
        '''
        return scores#, self.ans_attn(doc_hiddens, question_hidden, x1_mask)
