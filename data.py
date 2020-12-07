#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import random
import sys
import os
import re

import numpy as np

from io import open
from conllu import parse


UNK = '<<unk>>'
EOS = '<<eos>>'

FEATNONE = '#None'

RE_SPACES = re.compile(r'\s+')


def ud_treebank_path(base_path, treebank):
    return os.path.join(base_path, 'UD_' + treebank)


def ud_guess_base_name(path, ext):

    tb_files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for fn in filenames:
            if fn.endswith(ext):
                tb_files.append(fn)

    for fn in tb_files:
        base = os.path.basename(fn)
        mo = re.match('^(.*?)-train%s' % (ext), base)
        if mo:
            bn = mo.group(1)
            return bn

    return None


def load_conll_file(filename):
    text = open(filename, "r", encoding="utf-8").read()
    return parse(text)


def remove_spaces_in_forms(tb):
    for sent in tb:
        for tok in sent:
            tok['form'] = tok['form'].replace(' ', '')


def load_conll_treebank(base_path, treebank, load_dev_set=True):
    print('load_conll_treebank({}, {})'.format(base_path, treebank),
          file=sys.stderr)
    rv = {}

    treebank_path = ud_treebank_path(base_path, treebank)
    print('treebank_path: {}'.format(treebank_path), file=sys.stderr)
    base_name = ud_guess_base_name(treebank_path, '.conllu')
    print('base_name: {}'.format(base_name), file=sys.stderr)
    filename = os.path.join(treebank_path, base_name)
    print('filename: {}'.format(filename), file=sys.stderr)

    fn = filename + "-train.conllu"
    sys.stderr.write('Loading train set from %s ...\n' % (fn))
    rv["train"] = load_conll_file(fn)
    remove_spaces_in_forms(rv['train'])

    if load_dev_set:
        fn = filename + "-dev.conllu"
        sys.stderr.write('Loading dev set from %s ...\n' % (fn))
        rv["dev"]   = load_conll_file(fn)
        remove_spaces_in_forms(rv['dev'])

    for part in rv:
        for sent in rv[part]:
            for t in sent:
                form = RE_SPACES.sub('', t['form'])
                if form != t['form']:
                    sys.stderr.write('WARNING: [%s] spaces in form: \"%s\"\n' % (part, t['form']))
                    t['form'] = form

                if len(t['form']) == 0:
                    sys.stderr.write('WARNING: [%s] empty form is not allowed: \"%s\"\n' % (part, t['form']))
                    sys.stderr.write('Sentence: %s\n' % ' '.join([ x['form'] for x in sent ]))
                    raise

    #fn = filename + "-test.conllu"
    #sys.stderr.write('Loading test set from %s ...\n' % (fn))
    #rv["test"]  = load_conll_file(fn)

    return rv


def preprocess_deprel(s):
    return s


def collect_possible_tags(treebank):
    xpos = {}
    upos = {}
    rels = {}

    for k, v in treebank.items():
        for sent in v:
            for tok in sent:
                if not isinstance(tok['id'], int):
                    continue

                if tok['upostag'] is not None and tok['upostag'] not in upos:
                    upos[tok['upostag']] = True

                if tok['xpostag'] is not None and tok['xpostag'] not in xpos:
                    xpos[tok['xpostag']] = True

                deprel = preprocess_deprel(tok['deprel'])
                if deprel is not None and deprel not in rels:
                    rels[deprel] = True

    return list(upos.keys()), list(xpos.keys()), list(rels.keys())


def collect_possible_forms(treebank, parts, min_freq):
    forms = {}

    for k, v in treebank.items():
        if k not in parts:
            continue
        for sent in v:
            for tok in sent:
                form = tok['form'].lower()
                if form not in forms:
                    forms[form] = 1
                else:
                    forms[form] += 1

    cleaned_forms = []
    for f in forms:
        if forms[f] > min_freq:
            cleaned_forms.append(f)

    return cleaned_forms


def convert_feature_name(name):
    new_name = name.replace('[', '-_')
    new_name = new_name.replace(']', '_-')
    return new_name


def convert_feature_name_back(name):
    new_name = name.replace('-_', '[')
    new_name = new_name.replace('_-', ']')
    return new_name


def collect_possible_feats(treebank, parts):
    stat = {}
    total = 0

    for k, v in treebank.items():
        if k not in parts:
            continue
        for sent in v:
            for tok in sent:
                total += 1

                if tok['feats'] is None:
                    continue
                for name in tok['feats']:
                    value = tok['feats'][name]

                    name = convert_feature_name(name)

                    if name not in stat:
                        stat[name] = { }

                    if value not in stat[name]:
                        stat[name][value] = 1
                    else:
                        stat[name][value] += 1

    all_tokens = []

    for k, v in treebank.items():
        if k not in parts:
            continue
        for sent in v:
            for tok in sent:

                if tok['feats'] is None:
                    given_feats = {}
                else:
                    given_feats = tok['feats']

                all_feats = { k: FEATNONE for k in stat.keys() }

                for f in given_feats:
                    all_feats[convert_feature_name(f)] = given_feats[f]

                all_tokens.append(all_feats)

    feat_stat = {}
    for t in all_tokens:
        for f in t:
            v = t[f]
            if f not in feat_stat:
                feat_stat[f] = {}

            if v not in feat_stat[f]:
                feat_stat[f][v] = 1
            else:
                feat_stat[f][v] += 1

    none_freq = {}
    for f in sorted(feat_stat.keys()):
        l = "%10s:\t" % (f)
        for v in sorted(feat_stat[f].keys()):
            p = (feat_stat[f][v] * 1.0) / total
            if v == FEATNONE:
                none_freq[f] = p
            l += "%8s %.3f | " % (v, p)

        print(l)

    feats = {}
    for name in feat_stat:
        if none_freq[name] >= 0.95:
            continue
        feats[name] = []
        for value in feat_stat[name]:
            feats[name].append(value)

        feats[name].sort()

    return feats


def add_features_to_hp(hp, feats, all_treebanks, treebank_name):
    tasks = hp['tasks']
    for name in feats:
        full_name = treebank_name + '/feat/' + name
        if full_name in tasks:
            sys.stderr.write('Feat \'%s\' is already known' % (full_name))
            sys.exit(-1)

        i2t, t2i = build_word_indices(feats[name])
        tasks[full_name] = {
            'type': 'seqtag',
            'input': {
                'layer': 1,
                'direction': 'bi'
            },
            'i2t': i2t,
            't2i': t2i,
            'treebank': all_treebanks[treebank_name]
        }


def is_pseudographics(s):
    if len(s) < 20:
        return False
    pseudo_set = { x: 0 for x in r'-*+=|><_?!~{}\/' }
    for c in s:
        if c not in pseudo_set:
            return False
    return True


def build_char_indices(treebanks, parts, ignore_patterns=[]):
    c2i = {}
    max_word_len = 0

    for treebank_name in treebanks:
        for k, v in treebanks[treebank_name]['raw'].items():
            if k not in parts:
                continue
            for sent in v:
                for tok in sent:
                    ignore = False

                    for pattern in ignore_patterns:
                        if re.fullmatch(pattern, tok['form'], re.IGNORECASE):
                            ignore = True
                            break

                    if not ignore and len(tok['form']) > max_word_len:
                        if is_pseudographics(tok['form']):
                            ignore = True

                    if not ignore and len(tok['form']) > max_word_len:
                        if max_word_len > 50:
                            sys.stderr.write('WARNING: len==%d \"%s\"\n' % (max_word_len, tok['form']))
                            continue
                        max_word_len = len(tok['form'])


                    for c in tok['form']:
                        if c not in c2i:
                            c2i[c] = 1
                        else:
                            c2i[c] += 1

    i2c, c2i = build_word_indices(c2i.keys(), [EOS, UNK, ' '])
    return i2c, c2i, max_word_len + 2


def find_max_len(treebank, parts):
    max_len = 0

    for k, v in treebank.items():
        if k not in parts:
            continue
        for sent in v:
            if len(sent) > max_len:
                max_len = len(sent)

    return max_len


def add_special_tokens(wordlist):
    wordlist.append(UNK)


def build_word_indices(wordlist, first_items=None):
    t = list({ w: 0 for w in wordlist }.keys())
    t.sort()
    w2i = {}
    if first_items is not None:
        t = first_items + t

    for i in range(len(t)):
        if t[i] not in w2i:
            w2i[t[i]] = i

    i2w = sorted(w2i.keys(), key=lambda x: w2i[x])

    return i2w, w2i


def has_space_after(token):
    if token['misc'] is None:
        return True

    if 'SpaceAfter' in token['misc']:
        return token['misc']['SpaceAfter'] not in ['No', 'no']

    return True


def get_char_seq(sent, tidx, c2i, unk):
    word = sent[tidx-1]["form"]
    chars = [ c2i[c] if c in c2i else c2i[unk] for c in list(word)]
    #if len(sent) > tidx + 1:
    #    if has_space_after(sent[tidx]):
    #        chars.append(c2i[' '])
    #    next_word_start = [ c2i[c] if c in c2i else c2i[unk] for c in list(word[:1]) ]
    #    chars.extend(next_word_start)
    #else:
    #    chars.append(c2i[EOS])

    return chars


def generate_batch(tb, task, size, hp, start=0, flagRandom=True):
    if flagRandom:
        sent_indices = random.sample(range(len(tb)), size)
    else:
        sent_indices = range(len(tb))[start:start+size]

    word2idxp = hp['embd']['words'] #hp['input']['trainableDict']['w2i']
    word2idxt = hp['input']['trainableDict']['w2i']
    known_forms = hp['embd']['forms']
    char2idx = hp['input']['trainableDict']['c2i']

    b = { 'input_trainable': [], 'input_pretrained': [], 'chars': [], 'word_len': [], 'gold': [], 'len': [] }

    if len(task) > 0:
        cls2idx  = hp['tasks'][task]['t2i']
        task_name = task.split('/')[1].split('-')[0]

        if task.endswith('depparse'):
            b['gold_tags'] = []

    for idx in sent_indices:
        wt = [ 0 for i in range(hp['input']['maxSeqLen']) ] # for trainable
        wp = [ [ 0 for _ in range(hp['embdDim']) ] for i in range(hp['input']['maxSeqLen']) ] # for pretrained
        wl = [ 0 for i in range(hp['input']['maxSeqLen']) ]
        n = 1
        c = [ [ 0 for i in range(hp['input']['maxWordLen']) ] for i in range(hp['input']['maxSeqLen']) ]

        for t in tb[idx]:
            if isinstance(t['id'], tuple):
                continue

            form = t['form'].lower()

            # Training behavior
            if flagRandom:
                # Pretrained embeddings
                if t['form'] in word2idxp and ((not hp['embd']['allow_unk']) or (random.randint(0, 99) < 90)):
                    id = word2idxp[t['form']]
                    wp[n] = hp['embd']['vectors'][id]
                else:
                    if hp['embd']['allow_unk']:
                        if form in word2idxp and (random.randint(0, 99) < 90):
                            id = word2idxp[form]
                            wp[n] = hp['embd']['vectors'][id]
                        else:
                            id = word2idxp[UNK]
                            wp[n] = hp['embd']['vectors'][id]
                    else:
                        sys.stderr.write('ERROR: unknown words are not allowed: \"%s\"\n' % t['form'])
                        raise

                # Trainable embeddings
                if form in known_forms and (random.randint(0, 99) < 80):
                    wt[n] = word2idxt[form]
                else:
                    #sys.stderr.write('Form \'%s\' is considered UNK: %d %d\n' % (t['form'], int(form in known_forms), int(form in word2idxp)))
                    wt[n] = word2idxt[UNK]

            else:
                # Pretrained embeddings
                if t['form'] in word2idxp:
                    id = word2idxp[t['form']]
                    wp[n] = hp['embd']['vectors'][id]
                else:
                    if hp['embd']['allow_unk']:
                        if form in word2idxp:
                            id = word2idxp[form]
                            wp[n] = hp['embd']['vectors'][id]
                        else:
                            id = word2idxp[UNK]
                            wp[n] = hp['embd']['vectors'][id]
                    else:
                        sys.stderr.write('ERROR: Form \'%s\' is considered UNK\n' % (t['form']))
                        sys.stderr.write('Sentence: %s\n' % ' '.join([ x['form'] for x in tb[idx] ]))
                        raise # impossible with fastText

                # Trainable embeddings
                if form in known_forms:
                    wt[n] = word2idxt[form]
                else:
                    wt[n] = word2idxt[UNK]

            char_list = get_char_seq(tb[idx], n, char2idx, UNK)
            this_word_len = len(char_list)
            for i in range(min(this_word_len, hp['input']['maxWordLen'])):
                c[n][i] = char_list[i]
            wl[n] = min(this_word_len, hp['input']['maxWordLen'])
            #print("%s %s" % (t['form'], c[n]))
            n += 1

        if n >= hp['input']['maxSeqLen']:
            raise
        b['input_trainable'].append(wt)
        b['input_pretrained'].append(wp)
        b['len'].append(n)
        b['chars'].append(c)
        b['word_len'].append(wl)

        if len(task) == 0:
            continue

        if task_name.endswith('upos'):
            g = [0 for i in range(hp['input']['maxSeqLen'])]
            n = 1
            for t in tb[idx]:
                if isinstance(t['id'], tuple):
                    continue

                ci = cls2idx[t['upostag']]
                g[n] = ci
                n += 1

            if n != b['len'][-1]:
                raise

            b['gold'].append(g)

        elif task_name.endswith('xpos'):
            g = [0 for i in range(hp['input']['maxSeqLen'])]
            n = 1
            for t in tb[idx]:
                if isinstance(t['id'], tuple):
                    continue

                ci = cls2idx[t['xpostag']]
                g[n] = ci
                n += 1

            if n != b['len'][-1]:
                raise

            b['gold'].append(g)

        elif task_name.endswith('depparse'):
            # dep x head: 0 - no rel, 1 - rel exists
            # each line can have only one 1.
            g  = np.zeros([ hp['input']['maxSeqLen'] ]) #, hp['input']['maxSeqLen'] ])
            gt = np.zeros([ hp['input']['maxSeqLen'] ])
            n = 1
            for t in tb[idx]:
                if isinstance(t['id'], tuple):
                    continue

                head_idx = t['head']
                if '_' == head_idx or head_idx is None:
                    raise
                    head_idx = 0
                else:
                    head_idx = int(head_idx)

                if head_idx > hp['input']['maxSeqLen']:
                    raise
                g[n] = head_idx
                # TODO: try to transpose g. What is the difference?
                ci = cls2idx[preprocess_deprel(t['deprel'])]
                gt[n] = ci
                n += 1

            if n != b['len'][-1]:
                raise

            b['gold'].append(g)
            b['gold_tags'].append(gt)

        elif task_name.endswith('root'):
            g  = np.zeros([ hp['input']['maxSeqLen'] ])
            n = 1
            for t in tb[idx]:
                if isinstance(t['id'], tuple):
                    continue

                head_idx = t['head']
                if '_' == head_idx or head_idx is None:
                    head_idx = 0
                else:
                    head_idx = int(head_idx)

                if head_idx == 0:
                    g[n] = 1

                n += 1

            if n != b['len'][-1]:
                raise

            b['gold'].append(g)

        else:
            corpus_name, _, field_name = task.split('/')
            field_name = convert_feature_name_back(field_name)

            g = [0 for i in range(hp['input']['maxSeqLen'])]
            n = 1
            for t in tb[idx]:
                if isinstance(t['id'], tuple):
                    continue

                if t['feats'] is None or field_name not in t['feats']:
                    ci = cls2idx[FEATNONE]
                else:
                    ci = cls2idx[t['feats'][field_name]]
                g[n] = ci
                n += 1

            if n != b['len'][-1]:
                raise

            b['gold'].append(g)

    rv = {
        'input_trainable': np.array(b['input_trainable'], dtype=np.int32),
        'input_pretrained': np.array(b['input_pretrained'], dtype=np.float32),
        'chars': np.array(b['chars'], dtype=np.int32),
        'word_len': np.array(b['word_len'], dtype=np.int32),
        'len': np.array(b['len'], dtype=np.int32)
    }

    if len(task) == 0:
        return rv

    rv['gold'] = np.array(b['gold'], dtype=np.int32)

    if task.endswith('depparse'):
        rv['gold_tags'] = np.array(b['gold_tags'], dtype=np.int32)

    return rv


def evaluate_on_set(tb, task, hp, session, model):
    #shuffled = random.sample(tb, len(tb))

    all_predictions = []
    all_gold = []

    sentences_used = 0
    while len(tb) > sentences_used:
        batch_size = hp['batchSize']
        if len(tb) - sentences_used < batch_size:
            batch_size = len(tb) - sentences_used
        batch = generate_batch(tb, task, batch_size, hp, sentences_used, False)
        #batch = generate_batch(tb, task, len(tb), hp, 0, False)

        all_gold.extend(batch['gold'].tolist())

        sentences_used += hp['batchSize']

        pred = session.run(model['tasks'][task]['output']['pred'],
                           { model['input']['input_trainable']: batch['input_trainable'],
                             model['input']['input_pretrained']: batch['input_pretrained'],
                             model['input']['words']:     batch['chars'],
                             model['input']['word_len']:  batch['word_len'],
                             model['input']['len']:       batch['len'],
                             model['input']['batch_size']: batch_size
                             } )

        all_predictions.extend(pred.tolist())

    total = 0.0
    errors = 0.0
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    precision, recall, f1 = None, None, None

    for isent, sent in enumerate(all_predictions):
        for itok in range(1, len(tb[isent])):
            total += 1
            if all_predictions[isent][itok] != all_gold[isent][itok]:
                errors += 1

    accuracy = (total - errors) / total

    t2i = hp['tasks'][task]['t2i']
    if len(t2i) == 2:
        target = None
        for tag in t2i.keys():
            if tag != FEATNONE:
                target = tag

        if target is not None:
            for isent, sent in enumerate(all_predictions):
                for itok in range(1, len(tb[isent])):
                    if all_gold[isent][itok] == all_predictions[isent][itok]:
                        if all_predictions[isent][itok] == t2i[target]:
                            tp += 1
                        else:
                            tn += 1
                    else:
                        if all_predictions[isent][itok] == t2i[target]:
                            fp += 1
                        else:
                            fn += 1

            precision = (1.0 * tp) / (0.0001 + tp + fp)
            recall = (1.0 * tp) / (0.0001 + tp + fn)
            f1 = (2.0 * precision * recall) / (0.0001 + precision + recall)

    return accuracy, precision, recall, f1


def evaluate_parse_on_set(tb, task, hp, session, model):
    #shuffled = random.sample(tb, len(tb))

    all_predictions_arc = []
    all_predictions_rel = []

    all_gold_arc = []
    all_gold_rel = []

    sentences_used = 0
    while len(tb) > sentences_used:
        batch_size = hp['batchSize']
        if len(tb) - sentences_used < batch_size:
            batch_size = len(tb) - sentences_used
        batch = generate_batch(tb, task, batch_size, hp, sentences_used, False)
        #batch = generate_batch(tb, task, len(tb), hp, 0, False)

        all_gold_arc.extend(batch['gold'].tolist())
        all_gold_rel.extend(batch['gold_tags'].tolist())

        sentences_used += hp['batchSize']

        pred = session.run([model['tasks'][task]['arcs']['output']['pred'],
                            model['tasks'][task]['tags']['output']['pred']],
                                           #model[task]['metrics']['loss'],
                                           #model[task]['metrics']['acc'],
                                           #model[task]['metrics']['relloss'],
                                           #model[task]['metrics']['relacc']
                           { model['input']['input_trainable']: batch['input_trainable'],
                             model['input']['input_pretrained']: batch['input_pretrained'],
                             model['input']['words']:     batch['chars'],
                             model['input']['word_len']:  batch['word_len'],
                             model['input']['len']:       batch['len'],
                             model['input']['batch_size']: batch_size
                             } )

        #sys.stderr('%f %f %f %f\n' % (l, a, rl, ra))

        all_predictions_arc.extend(pred[0].tolist())
        all_predictions_rel.extend(pred[1].tolist())

    total = 0.0
    errors_arc = 0.0
    errors_rel = 0.0

    for isent, sent in enumerate(all_predictions_arc):
        for itok in range(1, len(tb[isent])):
            total += 1
            if all_predictions_arc[isent][itok] != all_gold_arc[isent][itok]:
                errors_arc += 1

            if all_predictions_rel[isent][itok] != all_gold_rel[isent][itok]:
                errors_rel += 1

    accuracy_arc = (total - errors_arc) / total
    accuracy_rel = (total - errors_rel) / total

    return accuracy_arc, accuracy_rel


def get_predictions(tb, task, batchSize, hp, session, model):
    shuffled = tb

    all_predictions = []
    all_gold = []

    sentences_used = 0
    while len(shuffled) - sentences_used > batchSize:
        batch = generate_batch(shuffled, task, hp, sentences_used, False)

        sentences_used += batchSize

        pred = session.run(model[task]['pred'],
                           { model['input']['sentences']: batch['input'],
                             model['input']['len']:       batch['len'] } )

        all_predictions.extend(pred.tolist())

    return all_predictions


def apply_predictions(tb, pred):
    modified_tb = tb

    sentn = 0
    for sent in modified_tb:
        tokn = 1
        for tok in sent:
            tok['head'] = int(pred[sentn][tokn])
            tokn += 1

        sentn += 1

        if sentn >= len(pred):
            break

    return modified_tb


def loadEmbeddings(fn):
    sys.stderr.write('Loading embeddings from %s ...\n' % (fn))
    f = open(fn, encoding='utf-8', mode='r')
    words = {}
    embd = []
    dim = None

    for line in f:
        splitline = line.rstrip().split(' ')
        word = splitline[0]
        try:
            embedding = [float(val) for val in splitline[1:]]
        except:
            raise
        words[word] = len(embd)
        embd.append(embedding)
        if dim is None:
            dim = len(embedding)
        else:
            if dim != len(embedding):
                raise

    if UNK not in words:
        words[UNK] = len(embd)
        embd.append([0] * len(embd[0]))
    if EOS not in words:
        words[EOS] = len(embd)
        embd.append([0] * len(embd[0]))

    return words, embd


def append_results(all, n):
    for k in n:
        if k not in all:
            if isinstance(n[k], dict):
                all[k] = {}
                append_results(all[k], n[k])
            else:
                all[k] = n[k].tolist()
        else:
            if isinstance(all[k], dict):
                append_results(all[k], n[k])
            else:
                all[k].extend(n[k].tolist())


def evaluate_seqtag(tb, hp, task, pred, gold):
    total = 0.0
    errors = 0.0
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    precision, recall, f1 = None, None, None

    for isent, sent in enumerate(pred):
        for itok in range(1, len(tb[isent])):
            total += 1
            if pred[isent][itok] != gold[isent][itok]:
                errors += 1

    accuracy = (total - errors) / total

    t2i = hp['tasks'][task]['t2i']
    if len(t2i) == 2:
        target = None
        for tag in t2i.keys():
            if tag != FEATNONE:
                target = tag

        if target is not None:
            for isent, sent in enumerate(pred):
                for itok in range(1, len(tb[isent])):
                    if gold[isent][itok] == pred[isent][itok]:
                        if pred[isent][itok] == t2i[target]:
                            tp += 1
                        else:
                            tn += 1
                    else:
                        if pred[isent][itok] == t2i[target]:
                            fp += 1
                        else:
                            fn += 1

            precision = (1.0 * tp) / (0.0001 + tp + fp)
            recall = (1.0 * tp) / (0.0001 + tp + fn)
            f1 = (2.0 * precision * recall) / (0.0001 + precision + recall)

    return accuracy, precision, recall, f1


def evaluate_depparse(tb, hp, task, pred, gold):
    total = 0.0
    errors_arc = 0.0
    errors_rel = 0.0

    for isent, sent in enumerate(pred['arcs']):
        for itok in range(1, len(tb[isent])):
            total += 1
            if pred['arcs'][isent][itok] != gold['arcs'][isent][itok]:
                errors_arc += 1

            if pred['tags'][isent][itok] != gold['tags'][isent][itok]:
                errors_rel += 1

    accuracy_arc = (total - errors_arc) / total
    accuracy_rel = (total - errors_rel) / total

    return accuracy_arc, accuracy_rel


def evaluate_all_tasks(treebanks, part, model, tasks, hp, sess, cache):

    all_results = {}
    all_gold = {}

    for treebank_name in treebanks:
        tb = treebanks[treebank_name]['raw'][part]

        fetches = {}

        for task_name in model['tasks']:
            if not task_name.startswith(treebank_name):
                continue

            m = model['tasks'][task_name]

            if task_name not in cache:
                cache[task_name] = { 'full': None, 'input_only': None }

            if cache[task_name]['full'] is None:
                cache[task_name]['full'] = generate_batch(tb, task_name, len(tb), hp, 0, False)

            batch = cache[task_name]['full']

            if m['type'] == 'seqtag':
                fetches[task_name] = \
                    {
                        'output': m['output']['pred']
                    }
                all_gold[task_name] = batch['gold']
            elif m['type'] == 'depparse':
                fetches[task_name] = \
                    {
                        'output':
                            {
                                'arcs': m['arcs']['output']['pred'],
                                'tags': m['tags']['output']['pred']
                            }
                    }
                all_gold[task_name] = \
                    {
                        'arcs': batch['gold'],
                        'tags': batch['gold_tags']
                    }

        if treebank_name not in cache:
            cache[treebank_name] = { 'full': None, 'input_only': None }

        if cache[treebank_name]['input_only'] is None:
            cache[treebank_name]['input_only'] = []

            sentences_used = 0
            while len(tb) > sentences_used:

                batch_size = hp['batchSize']

                if len(tb) - sentences_used < batch_size:
                    batch_size = len(tb) - sentences_used

                batch = generate_batch(tb, '', batch_size, hp, sentences_used, False)

                cache[treebank_name]['input_only'].append(batch)

                sentences_used += batch_size

        for batch in cache[treebank_name]['input_only']:
            results = sess.run(fetches,
                               {
                                   model['input']['input_trainable']:  batch['input_trainable'],
                                   model['input']['input_pretrained']: batch['input_pretrained'],
                                   model['input']['words']:            batch['chars'],
                                   model['input']['word_len']:         batch['word_len'],
                                   model['input']['len']:              batch['len'],
                                   model['input']['batch_size']:       len(batch['len'])
                                }
                               )

            append_results(all_results, results)

    for treebank_name in treebanks:
        tb = treebanks[treebank_name]['raw'][part]

        for task_name in model['tasks']:
            if not task_name.startswith(treebank_name):
                continue

            m = model['tasks'][task_name]

            if m['type'] == 'seqtag':
                accuracy, precision, recall, f1 = evaluate_seqtag(tb,
                                                                  hp,
                                                                  task_name,
                                                                  all_results[task_name]['output'],
                                                                  all_gold[task_name])
                tasks[task_name]['devacc'] = accuracy
                if precision is not None:
                    tasks[task_name]['precision'] = precision
                if recall is not None:
                    tasks[task_name]['recall'] = recall
                if f1 is not None:
                    tasks[task_name]['f1'] = f1

            elif m['type'] == 'depparse':
                accuracy_arcs, accuracy_tags = evaluate_depparse(tb,
                                                                 hp,
                                                                 task_name,
                                                                 all_results[task_name]['output'],
                                                                 all_gold[task_name])

                tasks[task_name]['arcs']['devacc'] = accuracy_arcs
                tasks[task_name]['tags']['devacc'] = accuracy_tags
