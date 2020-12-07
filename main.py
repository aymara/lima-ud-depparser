#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
import sys
import io
import time
import math
import argparse

import tensorflow as tf

from tensorflow.python.framework import graph_util

from data import *
from model import build_model, create_model_description
from utils import generate_output
from split_long_sentences import guess_max_sentence_len, augment
from languages import update_hp_for_corpus


UD_BASE_PATH = ''
EMBD_BASE_PATH = ''
OUT_FREQ = 10

HP = {
    'batchSize': 80,
    'embdDim': 300,
    'charEmbdDim': 16,
    'charLstmUnits': 32,
    'lstmUnits': 300,
    'lstmLayers': 4,
    'wordEmbdL2': 0.1,
    'charEmbdL2': 0.1,
    'learningRate': {
        'depparse': {
            'arcs': 0.001,
            'tags': 0.001
        },
        'seqtag': 0.001
    },
    'clipping': None,
    'dpSpecificLstm': {
        'layers': [ 200, 200 ]
    }
}

HP_small = {
    'batchSize': 80,
    'embdDim': 300,
    'charEmbdDim': 16,
    'charLstmUnits': 16,
    'lstmUnits': 100,
    'lstmLayers': 3,
    'wordEmbdL2': 0.1,
    'charEmbdL2': 0.1,
    'learningRate': {
        'depparse': {
            'arcs': 0.001,
            'tags': 0.001
        },
        'seqtag': 0.001
    },
    'clipping': None,
    'dpSpecificLstm': {
        'layers': [ 100, 100 ]
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU device for training')
    parser.add_argument('-d', '--device', type=int, help='Use GPU device with specified number')
    parser.add_argument('-t', '--treebank', help='Treebank name')
    parser.add_argument('-e', '--embeddings', help='Pretrained embeddings file name')
    parser.add_argument('-l', '--label', help='Label to tag output file')
    parser.add_argument('-p', '--prefix', help='Prefix output file')
    parser.add_argument('-u', '--udpath', help='UD corpus path')
    parser.add_argument('-m', '--max-epoch', type=int, help='Max number of epochs')
    args = parser.parse_args()

    global OUT_FREQ
    if args.gpu:
        gpu_to_use = "0"
        if args.device:
            gpu_to_use = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
        sys.stderr.write("Will use GPU %s\n" % (gpu_to_use))

        OUT_FREQ = 1000

    global UD_BASE_PATH, EMBD_BASE_PATH, HP
    HP = HP_small
    if args.udpath:
        UD_BASE_PATH = args.udpath

    main_treebank = args.treebank.split(',')[0]
    update_hp_for_corpus(HP, main_treebank)

    print(main_treebank)
    print(json.dumps(HP, indent=4))

    embd_words, embd_vectors = loadEmbeddings(args.embeddings)

    HP['embd'] = { 'words': embd_words, 'vectors': embd_vectors, 'allow_unk': False }

    forms = []

    max_len = 0
    guessed_max_sent_len = None

    treebanks = {}
    first_treebank_name = ''
    for treebank_name in args.treebank.split(','):
        if len(first_treebank_name) == 0:
            first_treebank_name = treebank_name

        treebanks[treebank_name] = {}
        tb = treebanks[treebank_name]
        tb['raw'] = load_conll_treebank(UD_BASE_PATH, treebank_name, True)
        tb['upostags'], tb['xpostags'], tb['reltypes'] = collect_possible_tags(tb['raw'])

        tb['upos_i2t'], tb['upos_t2i'] = build_word_indices(tb['upostags'])
        tb['xpos_i2t'], tb['xpos_t2i'] = build_word_indices(tb['xpostags'])
        tb['rel_i2t'], tb['rel_t2i'] = build_word_indices(tb['reltypes'])

        print('%s: found %d upostags, %d xpostags and %d reltypes' %
              (treebank_name,
               len(tb['upostags']),
               len(tb['xpostags']),
               len(tb['reltypes'])))

        forms.extend(collect_possible_forms(tb['raw'], ['train'], 7))

        if guessed_max_sent_len is None:
            guessed_max_sent_len = guess_max_sentence_len(tb['raw']['train'], 0.02)

        for p in tb['raw']:
            tb['raw'][p] = augment(tb['raw'][p], guessed_max_sent_len)

        _max_sent_len = find_max_len(tb['raw'], ['train', 'dev']) + 2
        if _max_sent_len > max_len:
            max_len = _max_sent_len

    print('Max sentence length is %d' % (max_len))

    forms_i2w, forms_w2i = build_word_indices(forms, [EOS, UNK])

    HP['embd']['forms'] = forms_i2w

    print('Found %d frequent forms (+EOS +UNK) in all trainsets (for trained embeddings)' % (len(forms_i2w)))

    i2c, c2i, max_word_len = build_char_indices(treebanks,
                                                ['train'],
                                                [
                                                    r'<?[a-z]{2,10}://[^ ]+',
                                                    r'www\.[^ ]+',
                                                    r'[^ ]+\.html[^ ]+',
                                                    r'[^ ]+\.jpg[^ ]+',
                                                    r'[^ ]+\.png[^ ]+',
                                                    r'<?[^@]+@[^@]+',
                                                    r'<?[a-z]+ns:[^ ]+',
                                                    r'[0-9]{20,}',
                                                    r'HKEY_[^ ]+',
                                                    r'[^ ]+////[^ ]+',
                                                    r'[^ ]+\\\\[^ ]+'
                                                ]
                                                )

    print('Max word length is %d' % (max_word_len))

    HP['input'] = { 'trainableDict': { 'i2c': i2c, 'c2i': c2i, 'i2w': forms_i2w, 'w2i': forms_w2i },
                     'maxSeqLen': max_len,
                     'maxWordLen': max_word_len }
    HP['tasks'] = {}

    for treebank_name in treebanks:
        tb = treebanks[treebank_name]

        if True:
            HP['tasks']['%s/%s' % (treebank_name, 'upos')] = {
                'type': 'seqtag',
                'treebank': tb,
                'i2t': tb['upos_i2t'],
                't2i': tb['upos_t2i'],
                'input': {'layer': 1, 'direction': 'bi'}
            }

        if False:
            HP['tasks']['%s/%s' % (treebank_name, 'xpos')] = {
                'type': 'seqtag',
                'treebank': tb,
                'i2t': tb['xpos_i2t'],
                't2i': tb['xpos_t2i'],
                'input': {'layer': 1, 'direction': 'bi'}
            }

        if False:
            HP['tasks']['%s/%s' % (treebank_name, 'upos-0-fw')] = HP['tasks']['%s/%s' % (treebank_name, 'upos')]
            HP['tasks']['%s/%s' % (treebank_name, 'upos-0-fw')]['input'] = {'layer': 0, 'direction': 'fw'}

            HP['tasks']['%s/%s' % (treebank_name, 'upos-0-bw')] = HP['tasks']['%s/%s' % (treebank_name, 'upos')]
            HP['tasks']['%s/%s' % (treebank_name, 'upos-0-bw')]['input'] = {'layer': 0, 'direction': 'bw'}

        if False:
            HP['tasks']['%s/%s' % (treebank_name, 'root')] = {
                'type': 'seqtag',
                'treebank': tb,
                'i2t': [ FEATNONE, 'root' ],
                't2i': { FEATNONE: 0, 'root': 1 },
                'input': {'layer': 1, 'direction': 'bi'}
            }

        HP['tasks']['%s/%s' % (treebank_name, 'depparse')] = {
            'type': 'depparse',
            'treebank': tb,
            'i2t': tb['rel_i2t'],
            't2i': tb['rel_t2i'],
            'dep_hidden_size': 256,
            'head_hidden_size': 256,
            'dep_output_size': 256,
            'head_output_size': 256
        }

    # collect existing features
    conll_feats = collect_possible_feats(treebanks[first_treebank_name]['raw'], ['train', 'dev'])

    # Add features to HP['tasks']
    add_features_to_hp(HP, conll_feats, treebanks, first_treebank_name)

    tf.reset_default_graph()

    model = build_model(HP, [ 'depparse' ])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    MODEL_FN = '%s%s' % (args.prefix, args.label)
    f = io.open('%s.conf' % MODEL_FN, 'w', encoding='utf-8')
    CONF_FOR_SAVE = create_model_description(HP, model)
    f.write(json.dumps(CONF_FOR_SAVE, ensure_ascii=False, indent=4))
    f.close()

    epoch = 0
    known_max_acc = 0.50
    eval_cache = {}
    curr = { 'history': [], 'best': {}, 'param': {} }

    for task_name in model['tasks']:
        if model['tasks'][task_name]['type'] == 'seqtag':
            curr['param'][task_name] = { 'lr': HP['learningRate']['seqtag'] }
            curr['best'][task_name] = {'devacc': 0.0}
        elif model['tasks'][task_name]['type'] == 'depparse':
            curr['param'][task_name + '/arcs'] = { 'lr': HP['learningRate']['depparse']['arcs'] }
            curr['param'][task_name + '/tags'] = { 'lr': HP['learningRate']['depparse']['tags'] }
            curr['best'][task_name + '/arcs'] = { 'devacc': 0.0 }
            curr['best'][task_name + '/tags'] = { 'devacc': 0.0 }

    stop = False
    while True:
        sys.stdout.write('EPOCH %6d ...\t\t\t\r' % (epoch))
        sys.stdout.flush()

        i = model['input']

        tasks = {}

        list_of_tasks = list(model['tasks'].keys())
        random.shuffle(list_of_tasks)

        before_iteration = time.time()

        for task_name in list_of_tasks:
            m = model['tasks'][task_name]

            skip_this_task = False
            if m['type'] == 'seqtag' and len(curr['history']) > 0 and task_name in curr['history'][-1] and curr['history'][-1][task_name]['acc'] > 0.99:
                sys.stderr.write('Skip training of %s\n' % (task_name))
                t = {'name': task_name, 'type': m['type'], 'acc': 0, 'loss': 0}
                skip_this_task = True

            if m['type'] == 'seqtag' and not skip_this_task:
                t = { 'name': task_name, 'type': m['type'] }
                batch = generate_batch(m['treebank']['raw']['train'], task_name, HP['batchSize'], HP)

                mo = m['output']
                o, t['loss'], t['acc'] = sess.run([mo['opt'],
                                                   mo['metrics']['loss'],
                                                   mo['metrics']['acc']
                                                   ],
                                                  {
                                                      i['input_trainable']:  batch['input_trainable'],
                                                      i['input_pretrained']: batch['input_pretrained'],
                                                      m['gold']:             batch['gold'],
                                                      i['len']:              batch['len'],
                                                      i['words']:            batch['chars'],
                                                      i['word_len']:         batch['word_len'],
                                                      i['keep_prob']:        0.66,
                                                      m['lr']:               curr['param'][task_name]['lr'],
                                                   })
                t['loss'] = t['loss'].item()

            elif m['type'] == 'depparse':
                t = { 'name': task_name, 'type': m['type'], 'arcs': {}, 'tags': {} }

                batch = generate_batch(m['treebank']['raw']['train'], task_name, HP['batchSize'], HP)

                gold = { 'arcs': m['arcs']['gold'], 'tags': m['tags']['gold'] }
                out_arcs, out_tags = m['arcs']['output'], m['tags']['output']
                all_outs = sess.run([out_arcs['opt'],
                                     out_tags['opt'],
                                     out_arcs['metrics']['loss'],
                                     out_arcs['metrics']['acc'],
                                     out_tags['metrics']['loss'],
                                     out_tags['metrics']['acc']
                                     ],
                                    {
                                        i['input_trainable']:  batch['input_trainable'],
                                        i['input_pretrained']: batch['input_pretrained'],
                                        gold['arcs']:          batch['gold'],
                                        gold['tags']:          batch['gold_tags'],
                                        i['len']:              batch['len'],
                                        i['words']:            batch['chars'],
                                        i['word_len']:         batch['word_len'],
                                        i['keep_prob']:        0.66,
                                        m['arcs']['lr']:       curr['param'][task_name + '/arcs']['lr'],
                                        m['tags']['lr']:       curr['param'][task_name + '/tags']['lr']
                                    })

                t['arcs']['loss'], t['arcs']['acc'], t['tags']['loss'], t['tags']['acc'] = all_outs[2:]
                t['arcs']['loss'] = t['arcs']['loss'].item()
                t['tags']['loss'] = t['tags']['loss'].item()

                if math.isnan(t['arcs']['loss']):
                    sys.stderr.write('ERROR: loss is NaN\n')
                    stop = True
                    break

                if math.isinf(t['arcs']['loss']):
                    sys.stderr.write('ERROR: loss is Inf\n')
                    stop = True
                    break

            tasks[task_name] = t

        after_iteration = time.time()
        sys.stderr.write('\t\tTraining (%d) took %d seconds\r' % (epoch, after_iteration - before_iteration))

        if stop:
            break

        if epoch > 0 and epoch % OUT_FREQ == 0:

            before = time.time()
            evaluate_all_tasks(treebanks, 'dev', model, tasks, HP, sess, eval_cache)
            after = time.time()
            sys.stderr.write('Evaluation took %d seconds\n' % (after - before))

            print(generate_output(epoch, tasks, args.treebank, curr))
            sys.stderr.flush()
            sys.stdout.flush()

            for task_name in model['tasks']:
                if model['tasks'][task_name]['type'] == 'seqtag':
                    if tasks[task_name]['devacc'] > curr['best'][task_name]['devacc']:
                        curr['best'][task_name]['devacc'] = tasks[task_name]['devacc']
                        curr['best'][task_name]['iter']   = iter
                    elif tasks[task_name]['devacc'] < curr['best'][task_name]['devacc']:
                        curr['param'][task_name]['lr'] = curr['param'][task_name]['lr'] * 0.99

                elif model['tasks'][task_name]['type'] == 'depparse':
                    for subtask in ['arcs', 'tags']:
                        if tasks[task_name][subtask]['devacc'] > curr['best'][task_name + '/' + subtask]['devacc']:
                            curr['best'][task_name + '/' + subtask]['devacc'] = tasks[task_name][subtask]['devacc']
                            curr['best'][task_name + '/' + subtask]['iter'] = iter
                        elif tasks[task_name][subtask]['devacc'] < curr['best'][task_name + '/' + subtask]['devacc']:
                            curr['param'][task_name + '/' + subtask]['lr'] = curr['param'][task_name + '/' + subtask]['lr'] * 0.995

            if tasks[first_treebank_name + '/depparse']['arcs']['devacc'] > (known_max_acc + 0.0001):

                # Save with variables as constants
                output_node_names = []
                for task in model['tasks'].keys():
                    if 'seqtag' == model['tasks'][task]['type']:
                        output_node_names.extend(model['tasks'][task]['nodes'])
                    elif 'depparse' == model['tasks'][task]['type']:
                        for subtask in model['tasks'][task].keys():
                            if 'nodes' in model['tasks'][task][subtask]:
                                output_node_names.extend(model['tasks'][task][subtask]['nodes'])

                output_graph_def = graph_util.convert_variables_to_constants(
                    sess,
                    tf.get_default_graph().as_graph_def(),
                    output_node_names
                )

                with tf.gfile.GFile('%s.model' % (MODEL_FN), 'wb') as f:
                    f.write(output_graph_def.SerializeToString())

                known_max_acc = tasks[first_treebank_name + '/depparse']['arcs']['devacc']

        curr['history'].append(tasks)

        main_task = '%s/%s/arcs' % (first_treebank_name, 'depparse')
        if curr['param'][main_task]['lr'] < 0.0001:
            sys.stderr.write('STOP at LR = %f\n' % (curr['param'][main_task]['lr']))
            break

        epoch += 1


if __name__ == '__main__':
    main()

