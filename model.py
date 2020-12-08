#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2018-2020 CEA LIST
#
# This file is part of LIMA.
#
# LIMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LIMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with LIMA.  If not, see <https://www.gnu.org/licenses/>

import numpy as np
import tensorflow as tf


def create_seqtag_output(out_name, opt, input, seq_len, gold, mask, total_words, lr):
    with tf.variable_scope('seqtag_' + out_name):
        # scorer
        logits = tf.layers.dense(input, len(opt['i2t']), name='%s_logits' % (out_name))

        # decoder
        crf = tf.get_variable("%s_crf" % (out_name), [len(opt['i2t']), len(opt['i2t'])], dtype=tf.float32)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, gold, seq_len, crf)

        # optimizer
        loss = tf.reduce_mean(-log_likelihood)
        optimizer = tf.train.AdamOptimizer(learning_rate = lr, beta2=0.9).minimize(loss)

        # predictions
        predictions, _ = tf.contrib.crf.crf_decode(logits, crf, seq_len)

        # metrics
        match = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predictions, gold), mask), tf.int32))
        accuracy = tf.truediv(match, total_words)

        return {
            'output': {
                'opt': optimizer,
                'metrics': {'loss': loss, 'acc': accuracy},
                'pred': predictions,
                'logits': logits,
                'names': {
                    'logits': logits.name.split(':')[0],
                    'crf': crf.name.split(':')[0]
                }
            },
            'nodes': [ predictions.name.split(':')[0], logits.name.split(':')[0], crf.name.split(':')[0] ]
        }


def create_depparse_output(out_name, opt, input, gold, mask, total_words, keep_prob,
                           embd_reg_loss, lr, batch_size, max_seq_len, clipping=None):
    with tf.variable_scope('depparse_' + out_name):
        # scorer
        dep_dense = tf.layers.dense(tf.layers.dropout(input, keep_prob),
                                    opt['dep_hidden_size'],
                                    activation = tf.nn.relu,
                                    name = '%s_arc_dep_hidden' % (out_name))

        head_dense = tf.layers.dense(tf.layers.dropout(input, keep_prob),
                                     opt['head_hidden_size'],
                                     activation = tf.nn.relu,
                                     name = '%s_arc_head_hidden' % (out_name))

        print("lstm_output.shape = " + str(input.shape))
        print("dep_dense.shape = " + str(dep_dense.shape))
        print("head_dense.shape = " + str(head_dense.shape))

        U1 = tf.get_variable("U1", [opt['head_hidden_size'], opt['head_hidden_size']], dtype=tf.float32)
        U1s = tf.tile(tf.expand_dims(U1, 0), [batch_size, 1, 1])
        print("U1s.shape = " + str(U1s.shape))
        HU1 = tf.matmul(head_dense, U1s)
        print("HU1.shape = " + str(HU1.shape))

        U2 = tf.get_variable("U2", [opt['head_hidden_size'], 1], dtype=tf.float32)
        U2s = tf.tile(U2, [1, max_seq_len])
        U2b = tf.tile(tf.expand_dims(U2s, 0), [batch_size, 1, 1])

        print('U2s.shape = ' + str(U2s.shape))
        print('U2b.shape = ' + str(U2b.shape))

        #arc_logits = tf.matmul(HU1, dep_dense, transpose_b=True) + tf.matmul(head_dense, U2b)
        arc_logits = tf.add(tf.matmul(HU1, dep_dense, transpose_b=True), tf.matmul(head_dense, U2b), name='%s_arc_logits' % (out_name))
        print('arc_logits.shape = ' + str(arc_logits.shape))
        print('gold[\'arcs\'].shape = ' + str(gold['arcs'].shape))
        print('mask.shape = ' + str(mask.shape))
        #depparse_probs = tf.nn.softmax(depparse_logits)
        arc_preds = tf.to_int32(tf.argmax(arc_logits, axis=-1), name='%s_arc_preds' % (out_name))
        print('arc_preds.shape = ' + str(arc_preds.shape))
        #print('arc_preds.name = ' + str(arc_preds.name.split(':')[0]))

        arc_ce = tf.losses.sparse_softmax_cross_entropy(logits = arc_logits,
                                                        labels = gold['arcs'],
                                                        weights = mask)
        arc_loss = tf.reduce_sum(arc_ce) + embd_reg_loss

        # gradient clipping
        if clipping is not None:
            optimizer = tf.train.AdamOptimizer(learning_rate=lr[0], beta2=0.9)

            gradient_var_pairs = optimizer.compute_gradients(arc_loss)
            vars = [x[1] for x in gradient_var_pairs]
            gradients = [x[0] for x in gradient_var_pairs]
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clipping)
            arc_optimizer = optimizer.apply_gradients(zip(clipped_gradients, vars))
        else:
            arc_optimizer = tf.train.AdamOptimizer(learning_rate = lr[0], beta2=0.9).minimize(arc_loss)

        arc_match = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(arc_preds, gold['arcs']), mask), tf.int32))
        arc_acc = tf.truediv(arc_match, total_words)

        # scorer (ARC LABELS)
        Z1 = tf.get_variable("Z1", [ opt['head_hidden_size'], len(opt['i2t']), opt['head_hidden_size'] ], dtype=tf.float32)
        Z1s = tf.tile(tf.expand_dims(Z1, 0), [batch_size, 1, 1, 1])
        print("Z1s.shape = " + str(Z1s.shape))

        # [batch_size, d, c, d] => [batch_size, d, c * d]
        Z1sr = tf.reshape(Z1s, [ -1, opt['head_hidden_size'], len(opt['i2t']) * opt['head_hidden_size'] ])
        print("Z1sr.shape = " + str(Z1sr.shape))

        # [batch_size, l, d] x [batch_size, d, c * d] = [batch_size, l, c * d]
        HZ1 = tf.matmul(head_dense, Z1sr)
        print("HZ1.shape = " + str(HZ1.shape))

        # [batch_size, l, c * d] => [batch_size, l * c, d]
        HZ1r = tf.reshape(HZ1, [ -1, HZ1.shape[1] * len(opt['i2t']), opt['head_hidden_size'] ])
        print("HZ1r.shape = " + str(HZ1r.shape))

        # [batch_size, l * c, d] x [batch_size, l, d]T => [batch_size, l * c, d] x [batch_size, d, l] = [batch_size, l * c, l]
        HZ1D = tf.matmul(HZ1r, dep_dense, transpose_b=True)
        print("HZ1D.shape = " + str(HZ1D.shape))

        # [batch_size, l * c, l] => [batch_size, l, c, l]
        HZ1Dr = tf.reshape(HZ1D, [ -1, HZ1D.shape[2], len(opt['i2t']), HZ1D.shape[2] ])
        print("HZ1Dr.shape = " + str(HZ1Dr.shape))

        # TODO: missing part is here
        W1 = tf.get_variable("W1", [ opt['head_hidden_size'], len(opt['i2t']) ], dtype=tf.float32)
        HW1 = tf.matmul(head_dense, tf.tile(tf.expand_dims(W1, 0), [batch_size, 1, 1]) )

        W2 = tf.get_variable("W2", [opt['head_hidden_size'], len(opt['i2t'])], dtype=tf.float32)
        DW2 = tf.matmul(dep_dense, tf.tile(tf.expand_dims(W2, 0), [batch_size, 1, 1]))

        b = tf.get_variable("b", [len(opt['i2t']), 1], dtype=tf.float32)

        Y = HZ1Dr + tf.expand_dims(HW1, -1) + tf.expand_dims(DW2, -1) + b

        one_hot = tf.one_hot(arc_preds, arc_preds.shape[1])
        print("one_hot.shape = " + str(one_hot.shape))

        one_hot = tf.expand_dims(one_hot, 3)
        print("one_hot.shape = " + str(one_hot.shape))

        rel_logits = tf.matmul(Y, one_hot)
        print("rel_logits.shape = " + str(rel_logits.shape))

        rel_logits = tf.squeeze(rel_logits, axis=3, name='%s_rel_logits' % (out_name))
        print("rel_logits.shape = " + str(rel_logits.shape))

        rel_preds = tf.to_int32(tf.argmax(rel_logits, axis=-1), name='%s_rel_preds' % (out_name))
        print("rel_preds.shape = " + str(rel_preds.shape))

        rel_match = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(rel_preds, gold['tags']), mask), tf.int32))
        rel_acc = tf.truediv(rel_match, total_words)

        rel_ce = tf.losses.sparse_softmax_cross_entropy(logits = rel_logits,
                                                        labels = gold['tags'],
                                                        weights = mask)
        rel_loss = tf.reduce_sum(rel_ce)

        rel_optimizer = tf.train.AdamOptimizer(learning_rate = lr[1], beta2 = 0.9).minimize(rel_loss)

        tasks = {}
        tasks['arcs'] = {
            'gold': gold['arcs'],
            'output':
                {
                    'opt': arc_optimizer,
                    'pred': arc_preds,
                    'logits': arc_logits,
                    'metrics':
                        {
                            'acc': arc_acc,
                            'loss': arc_loss
                        },
                    'names': {
                        'pred': arc_preds.name.split(':')[0],
                        'logits': arc_logits.name.split(':')[0]
                    }
                },
            'nodes': [ arc_preds.name.split(':')[0], arc_logits.name.split(':')[0] ]
        }
        tasks['tags'] = {
            'gold': gold['tags'],
            'output':
                {
                    'opt': rel_optimizer,
                    'pred': rel_preds,
                    'logits': rel_logits,
                    'metrics':
                        {
                            'acc': rel_acc,
                            'loss': rel_loss
                        },
                    'names': {
                        'pred': rel_preds.name.split(':')[0],
                        'logits': rel_logits.name.split(':')[0]
                    }
                },
            'nodes': [ rel_preds.name.split(':')[0], rel_logits.name.split(':')[0] ]
        }

        return tasks


def create_rnn(input, seq_len, units, keep_prob, scope_prefix):
    rnn_input = tf.contrib.rnn.transpose_batch_time(input)
    all_rnn_outputs = []

    num_layers = len(units)
    for n in range(num_layers):
        rnn_fw_cell = tf.contrib.rnn.LSTMBlockCell(units[n])
        rnn_bw_cell = tf.contrib.rnn.LSTMBlockCell(units[n])

        rnn_fw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_cell,
                                                     input_keep_prob = keep_prob,
                                                     state_keep_prob = keep_prob,
                                                     variational_recurrent=True,
                                                     input_size=rnn_input.shape[-1],
                                                     dtype=rnn_input.dtype)
        rnn_bw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_cell,
                                                     input_keep_prob=keep_prob,
                                                     state_keep_prob=keep_prob,
                                                     variational_recurrent=True,
                                                     input_size=rnn_input.shape[-1],
                                                     dtype=rnn_input.dtype)

        values, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_fw_cell,
                                                    cell_bw=rnn_bw_cell,
                                                    inputs=rnn_input,
                                                    sequence_length=seq_len,
                                                    scope=('%s_%s' % (scope_prefix, n)),
                                                    dtype=tf.float32,
                                                    time_major=True)
        rnn_output = tf.concat(values, 2)
        rnn_input = rnn_output

        this_layer = {
            'fw': tf.contrib.rnn.transpose_batch_time(values[0]),
            'bw': tf.contrib.rnn.transpose_batch_time(values[1]),
            'bi': tf.contrib.rnn.transpose_batch_time(rnn_output)
        }

        all_rnn_outputs.append(this_layer)

    return all_rnn_outputs


def create_fused_rnn(input, seq_len, units, keep_prob, scope_prefix):
    with tf.variable_scope(scope_prefix):
        rnn_input = tf.contrib.rnn.transpose_batch_time(input)
        all_rnn_outputs = []

        num_layers = len(units)
        for n in range(num_layers):
            rnn_input = tf.layers.dropout(rnn_input, keep_prob)

            rnn_fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(units[n])
            rnn_bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(units[n])

            # rnn_fw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_cell,
            #                                              input_keep_prob = keep_prob,
            #                                              state_keep_prob = keep_prob,
            #                                              variational_recurrent=True,
            #                                              input_size=rnn_input.shape[-1],
            #                                              dtype=rnn_input.dtype)
            # rnn_bw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_cell,
            #                                              input_keep_prob=keep_prob,
            #                                              state_keep_prob=keep_prob,
            #                                              variational_recurrent=True,
            #                                              input_size=rnn_input.shape[-1],
            #                                              dtype=rnn_input.dtype)
            #
            # values, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_fw_cell,
            #                                             cell_bw=rnn_bw_cell,
            #                                             inputs=rnn_input,
            #                                             sequence_length=seq_len,
            #                                             scope=('%s_%s' % (scope_prefix, n)),
            #                                             dtype=tf.float32)

            fw_output, fw_state = rnn_fw_cell(rnn_input, dtype=tf.float32)
            bw_input = tf.reverse_sequence(rnn_input, seq_len, seq_dim=0, batch_dim=1)
            bw_output, bw_state = rnn_bw_cell(bw_input, dtype=tf.float32)
            bw_output = tf.reverse_sequence(bw_output, seq_len, seq_dim=0, batch_dim=1)

            rnn_output = tf.concat([fw_output, bw_output], 2)
            rnn_input = rnn_output

            this_layer = {
                'state': {
                    'fw': fw_state.h, #tf.concat([fw_state.c, fw_state.h], 1),
                    'bw': bw_state.h, #tf.concat([bw_state.c, bw_state.h], 1)
                },
                'fw': tf.contrib.rnn.transpose_batch_time(fw_output),
                'bw': tf.contrib.rnn.transpose_batch_time(bw_output),
                'bi': tf.contrib.rnn.transpose_batch_time(rnn_output)
            }

            all_rnn_outputs.append(this_layer)

        return all_rnn_outputs


def build_model(hp, tasks):

    # input
    pretrainedVectors = tf.get_variable(name='pretrained_vectors',
                                        shape=[len(hp['embd']['vectors']), hp['embdDim']],
                                        trainable=False,
                                        #initializer=tf.constant_initializer(np.zeros((len(hp['embd']['vectors']), hp['embdDim']))))
                                        initializer=tf.constant_initializer(np.array(hp['embd']['vectors'])))

    wordVectors = tf.get_variable(name="word_vectors",
                                  shape=[len(hp['embd']['forms']), hp['embdDim']])
                                  #initializer=tf.random_uniform_initializer(-0.001, 0.001))
    charVectors = tf.get_variable(name="char_vectors",
                                  shape=[len(hp['input']['trainableDict']['i2c']), hp['charEmbdDim']])
                                  #initializer=tf.random_uniform_initializer(-0.04, 0.04))

    print("pretrainedVectors.shape = " + str(pretrainedVectors.shape))
    print("wordVectors.shape = " + str(wordVectors.shape))

    input_trainable = tf.placeholder(tf.int32, [None, hp['input']['maxSeqLen']], name='input_trainable')
    #input_pretrained = tf.placeholder(tf.int32, [None, hp['input']['maxSeqLen']], name='input_pretrained')
    input_pretrained = tf.placeholder(tf.float32, [None, hp['input']['maxSeqLen'], hp['embdDim']], name='input_pretrained')

    seq_len = tf.placeholder(tf.int32, [None], name="len")
    word_len = tf.placeholder(tf.int32, [None, hp['input']['maxSeqLen']], name="word_len")
    mask = tf.sequence_mask(seq_len, hp['input']['maxSeqLen'])
    keep_prob = tf.placeholder_with_default(1.0, [], name='keep_prob')
    batchSize = tf.placeholder_with_default(hp['batchSize'], [], name='batch_size')
    total_words = tf.reduce_sum(seq_len)

    input_trainable_word_ids, _ = tf.unique(tf.reshape(input_trainable, [-1]))
    print("input_trainable_word_ids.shape = " + str(input_trainable_word_ids.shape))
    input_trainable_word_embd_slice = tf.gather(wordVectors, input_trainable_word_ids)
    print("input_trainable_word_embd_slice.shape = " + str(input_trainable_word_embd_slice.shape))
    input_trainable_word_reg_loss = tf.nn.l2_loss(input_trainable_word_embd_slice) * hp['wordEmbdL2']

    char_input_data = tf.placeholder(tf.int32, [None, hp['input']['maxSeqLen'], hp['input']['maxWordLen']], name='words')
    char_embd = tf.nn.embedding_lookup(charVectors, char_input_data)

    input_trainable_char_ids, _ = tf.unique(tf.reshape(char_input_data, [-1]))
    print("input_trainable_char_ids.shape = " + str(input_trainable_char_ids.shape))
    input_trainable_char_embd_slice = tf.gather(charVectors, input_trainable_char_ids)
    print("input_trainable_char_embd_slice.shape = " + str(input_trainable_char_embd_slice.shape))
    input_trainable_char_reg_loss = tf.nn.l2_loss(input_trainable_char_embd_slice) * hp['charEmbdL2']

    embd_reg_loss = input_trainable_word_reg_loss + input_trainable_char_reg_loss

    #batchSize = hp['batchSize']

    print("char_embd.shape = " + str(char_embd.shape))
    char_embd = tf.reshape(char_embd, shape=[ -1, hp['input']['maxWordLen'], hp['charEmbdDim']])
    print("char_embd.shape = " + str(char_embd.shape))

    word_len_reshaped = tf.reshape(word_len, shape=[-1])

    char_lstm_fw_cell = tf.contrib.rnn.LSTMBlockCell(hp['charLstmUnits'])
    char_lstm_bw_cell = tf.contrib.rnn.LSTMBlockCell(hp['charLstmUnits'])
    _, ((_, char_state_fw), (_, char_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=tf.nn.rnn_cell.DropoutWrapper(char_lstm_fw_cell,
                                                input_keep_prob=keep_prob,
                                                state_keep_prob=keep_prob,
                                                variational_recurrent=True,
                                                input_size=char_embd.shape[-1],
                                                dtype=char_embd.dtype),
          cell_bw=tf.nn.rnn_cell.DropoutWrapper(char_lstm_bw_cell,
                                                input_keep_prob=keep_prob,
                                                state_keep_prob=keep_prob,
                                                variational_recurrent=True,
                                                input_size=char_embd.shape[-1],
                                                dtype=char_embd.dtype),
          inputs=char_embd,
          sequence_length=word_len_reshaped,
          scope=('charBLSTM_1'),
          dtype=tf.float32)

    #char_rnn = create_fused_rnn(char_embd, word_len_reshaped, [ hp['charLstmUnits'] ], keep_prob, 'CharRNN')
    #char_state_fw = char_rnn[-1]['state']['fw']
    #char_state_bw = char_rnn[-1]['state']['bw']

    print("char_state_fw.shape = " + str(char_state_fw.shape))
    char_representation = tf.concat((char_state_fw, char_state_bw), -1)
    char_representation = tf.reshape(char_representation, shape=[ -1, hp['input']['maxSeqLen'], 2*hp['charLstmUnits'] ])
    print("char_representation.shape = " + str(char_representation.shape))

    #pretrainedEmbd = tf.nn.embedding_lookup(pretrainedVectors, input_pretrained)
    trainedEmbd = tf.nn.embedding_lookup(wordVectors, input_trainable)
    print("input_pretrained.shape = " + str(input_pretrained.shape))
    print("trainedEmbd.shape = " + str(trainedEmbd.shape))
    data = tf.add(input_pretrained, trainedEmbd)
    #print("pretrainedEmbd.shape = " + str(pretrainedEmbd.shape))
    print("data.shape = " + str(data.shape))

    # encoder
    lstm_input = tf.concat([data, char_representation], 2)
    print("lstm_input.shape = " + str(lstm_input.shape))
    #all_lstm_outputs = create_fused_rnn(lstm_input, seq_len, [hp['lstmUnits']] * hp['lstmLayers'], keep_prob, 'MainRnn')
    all_lstm_outputs = create_rnn(lstm_input, seq_len, [hp['lstmUnits']] * hp['lstmLayers'], keep_prob, 'MainRnn')

    dp_specific_rnn = None
    if 'dpSpecificLstm' in hp and hp['dpSpecificLstm'] is not None:
        #dp_specific_rnn = create_fused_rnn(lstm_input, seq_len, hp['dpSpecificLstm']['layers'], keep_prob, 'DpRnn')
        dp_specific_rnn = create_rnn(lstm_input, seq_len, hp['dpSpecificLstm']['layers'], keep_prob, 'DpRnn')

    # All sequence taggers start here
    all_features = {}

    for task_name in hp['tasks']:
        if 'type' not in hp['tasks'][task_name] or hp['tasks'][task_name]['type'] != 'seqtag':
            continue

        task_gold = tf.placeholder(tf.int32,
                                      [ hp['batchSize'], hp['input']['maxSeqLen'] ],
                                      name = "gold_%s" % (task_name))
        lr = tf.placeholder_with_default(0.001, [])

        if 'input' in hp['tasks'][task_name]:
            num_layer = hp['tasks'][task_name]['input']['layer']
            direction = hp['tasks'][task_name]['input']['direction']
            this_task_input = all_lstm_outputs[num_layer][direction]
        else:
            this_task_input = all_lstm_outputs[-1]['bi']

        task = create_seqtag_output(task_name,
                                    hp['tasks'][task_name],
                                              tf.layers.dropout(this_task_input, keep_prob),
                                              seq_len,
                                              task_gold,
                                              mask,
                                              total_words,
                                              lr)

        task['gold'] = task_gold
        task['type'] = hp['tasks'][task_name]['type']
        task['treebank'] = hp['tasks'][task_name]['treebank']
        task['lr'] = lr

        all_features[task_name] = task

    # Depparse tasks
    for task_name in hp['tasks']:
        if 'type' not in hp['tasks'][task_name] or hp['tasks'][task_name]['type'] != 'depparse':
            continue

        task_gold = {}
        task_gold['arcs'] = tf.placeholder(tf.int32,
                                       [None, hp['input']['maxSeqLen']],
                                       name = 'gold_arcs_%s' % (task_name))
        task_gold['tags'] = tf.placeholder(tf.int32,
                                       [None, hp['input']['maxSeqLen']],
                                       name = 'gold_tags_%s' % (task_name))
        lr_arcs = tf.placeholder_with_default(0.001, [])
        lr_tags = tf.placeholder_with_default(0.001, [])


        if dp_specific_rnn is not None:
            this_task_input = tf.concat([all_lstm_outputs[-1]['bi'], dp_specific_rnn[-1]['bi']], 2)
        else:
            this_task_input = all_lstm_outputs[-1]['bi']

        print("this_task_input.shape = " + str(this_task_input.shape))

        task = create_depparse_output(task_name,
                                      hp['tasks'][task_name],
                                      this_task_input,
                                      task_gold,
                                      mask,
                                      total_words,
                                      keep_prob,
                                      embd_reg_loss,
                                      [lr_arcs, lr_tags],
                                      batchSize,
                                      hp['input']['maxSeqLen'],
                                      hp['clipping'])

        task['type'] = hp['tasks'][task_name]['type']
        task['treebank'] = hp['tasks'][task_name]['treebank']
        task['arcs']['lr'] = lr_arcs
        task['tags']['lr'] = lr_tags

        all_features[task_name] = task

    return { 'input':    { 'input_trainable': input_trainable,
                           'input_pretrained': input_pretrained,
                           'len': seq_len,
                           'words': char_input_data,
                           'word_len': word_len,
                           'keep_prob': keep_prob,
                           'batch_size': batchSize
                           },
             'tasks':     all_features
            }


def create_model_description(conf, model):
    descr = {
        'conf': {
            'batchSize':  conf['batchSize'],
            'maxSeqLen':  conf['input']['maxSeqLen'],
            'maxWordLen': conf['input']['maxWordLen']
        },
        'dicts': {
            'c2i': conf['input']['trainableDict']['c2i'],
            'w2i': conf['input']['trainableDict']['w2i']
        },
        'input': {
            'input_trainable':  model['input']['input_trainable'].name.split(':')[0],
            'input_pretrained': model['input']['input_pretrained'].name.split(':')[0],
            'len':              model['input']['len'].name.split(':')[0],
            'words':            model['input']['words'].name.split(':')[0],
            'word_len':         model['input']['word_len'].name.split(':')[0],
            'batch_size':       model['input']['batch_size'].name.split(':')[0]
        },
        'output':    {},
    }

    for task in conf['tasks']:
        descr['output'][task] = {
            'type': conf['tasks'][task]['type']
        }

        if descr['output'][task]['type'] == 'seqtag':
            descr['output'][task]['i2t'] = conf['tasks'][task]['i2t']
            descr['output'][task]['nodes'] = model['tasks'][task]['output']['names']
        elif descr['output'][task]['type'] == 'depparse':
            descr['output'][task]['arcs'] = {
                'nodes': model['tasks'][task]['arcs']['output']['names']
            }
            descr['output'][task]['tags'] = {
                'nodes': model['tasks'][task]['tags']['output']['names'],
                'i2t': model['tasks'][task]['treebank']['rel_i2t']
            }

    return descr