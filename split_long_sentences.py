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

import sys
import math

from io import open
from conllu import parse


def load_conll_file(filename):
    text = open(filename, "r", encoding="utf-8").read()
    return parse(text)


def save_tb(tb):
    for sent in tb:
        if sent.metadata:
            for key, value in sent.metadata.items():
                if value is None:
                    sent.metadata[key] = 'None'

        print(sent.serialize(), end='')


def split_sentence_impl(sent):
    ret = []

    last_idx = -1
    last_word_id = sent[last_idx]['id']
    if not isinstance(last_word_id, int):
        last_idx -= 1
        last_word_id = sent[last_idx]['id']

    possible_split_points = {}

    for token in sent:
        if not isinstance(token['id'], int):
            continue

        pos = token['id']
        # consider splitting sentence after token pos

        if pos < 3 or pos > last_word_id - 3:
            continue

        links_to_update_before = 0
        links_to_update_after = 0

        for tok in sent:
            if not isinstance(tok['id'], int):
                continue

            if tok['id'] <= pos:
                if tok['head'] > pos:
                    links_to_update_before += 1
            else:
                if tok['head'] <= pos:
                    links_to_update_after += 1

        possible_split_points[pos] = links_to_update_before + links_to_update_after

    good_split_points = sorted(possible_split_points.keys(), key=lambda x: possible_split_points[x])
    min_value = possible_split_points[good_split_points[0]]
    min_values_count = sum(v == min_value for v in possible_split_points.values())
    good_split_points = good_split_points[:min_values_count]

    if len(good_split_points) == 0:
        raise

    good_split_points_centred = sorted(good_split_points, key=lambda x: abs(int(math.floor(len(sent) / 2)) - x))

    if len(good_split_points_centred) == 0:
        raise

    split_pos = good_split_points_centred[0]
    split_idx = None

    if split_pos is None:
        raise

    for i in range(len(sent)):
        if isinstance(sent[i]['id'], int) and sent[i]['id'] == split_pos:
            split_idx = i + 1
            break

    first = sent.copy()
    first.metadata = first.metadata.copy()
    first.tokens = first[:split_idx]
    if 'sent_id' in first.metadata:
        first.metadata['sent_id'] = first.metadata['sent_id'] + " / left"

    update_heads_in_left_part(first, sent)

    for tok in first:
        if tok['head'] is None:
            continue
        if tok['head'] != 0 and tok['head'] > split_pos:
            sys.stderr.write('F prev head of token %s is %d\n' % (str(tok['id']), tok['head']))

    second = sent.copy()
    second.metadata = second.metadata.copy()
    second.tokens = second[split_idx:]
    if 'sent_id' in second.metadata:
        second.metadata['sent_id'] = second.metadata['sent_id'] + " / right"

    update_heads_in_right_part(second, sent)

    i = len(second.tokens) - 1
    last_token_id = second[i]['id']
    while not isinstance(last_token_id, int):
        i -= 1
        last_token_id = second[i]['id']

    for tok in second:
        if 'head' not in tok or tok['head'] is None or tok['head'] == '_':
            continue
        #if tok['head'] != 0 and tok['head'] > last_token_id:
        #    sys.stderr.write('S prev head of token %s is %d\n' % (str(tok['id']), tok['head']))

    ret = [first, second]

    return ret


def get_token_with_id(sent, id):
    for tok in sent:
        if str(tok['id']) == id:
            return tok
    raise


def update_heads_in_left_part(part, whole_sent):
    i = len(part) - 1
    last_token_id = part[i]['id']
    while not isinstance(last_token_id, int):
        i -= 1
        last_token_id = part[i]['id']

    root_pos = None
    orphans = []
    for i in range(len(part)):
        if 'head' not in part[i] or part[i]['head'] is None or part[i]['head'] == '_':
            continue
        if part[i]['head'] == 0:
            root_pos = part[i]['id']
        if part[i]['head'] > last_token_id:
            orphans.append(part[i]['id'])

    if root_pos is None:
        branch_len = [ len(part) + 1 ] * len(part)
        for i in range(len(part)):
            #pos = orphans[i]
            if not isinstance(part[i]['id'], int):
                continue
            if 'upostag' not in part[i]:
                continue
            if part[i]['upostag'] == 'PUNCT':
                continue
            token = get_token_with_id(whole_sent, str(part[i]['id']))
            parent = token['head']
            branch_len[i] = 0
            while parent != 0:
                if branch_len[i] > len(whole_sent) + 1:
                    raise
                branch_len[i] += 1
                parent = get_token_with_id(whole_sent, str(parent))['head']

        root_pos = part[branch_len.index(min(branch_len))]['id']
        get_token_with_id(part, str(root_pos))['head'] = 0

    for tok in part:
        if 'head' not in tok or tok['head'] is None or tok['head'] == '_':
            continue
        if tok['head'] != 0 and tok['head'] > last_token_id:
            tok['head'] = root_pos

    update_indices(part)


def update_heads_in_right_part(part, whole_sent):
    i = 0
    first_token_id = part[i]['id']
    while not isinstance(first_token_id, int):
        i += 1
        first_token_id = part[i]['id']

    root_pos = None
    orphans = []
    for i in range(len(part)):
        if 'head' not in part[i] or part[i]['head'] is None or part[i]['head'] == '_':
            continue
        if part[i]['head'] == 0:
            root_pos = part[i]['id']
        if part[i]['head'] < first_token_id:
            orphans.append(part[i]['id'])

    if root_pos is None:
        branch_len = [ len(part) + 1 ] * len(part)
        for i in range(len(part)):
            #pos = orphans[i]
            if not isinstance(part[i]['id'], int):
                continue
            if 'upostag' not in part[i]:
                continue
            if part[i]['upostag'] == 'PUNCT':
                continue
            token = get_token_with_id(whole_sent, str(part[i]['id']))
            parent = token['head']
            branch_len[i] = 0
            while parent != 0:
                if branch_len[i] > len(whole_sent) + 1:
                    raise
                branch_len[i] += 1
                parent = get_token_with_id(whole_sent, str(parent))['head']

        root_pos = part[branch_len.index(min(branch_len))]['id']
        get_token_with_id(part, str(root_pos))['head'] = 0

    for tok in part:
        if 'head' not in tok or tok['head'] is None or tok['head'] == '_':
            continue
        if tok['head'] != 0 and tok['head'] < first_token_id:
            tok['head'] = root_pos

    update_indices(part)


def update_indices(sent):
    pos = 1
    old2new = {}

    for i in range(len(sent)):
        id = sent[i]['id']
        if id not in old2new:
            if isinstance(id, int):
                new_id = pos
                old2new[id] = pos
                pos += 1
            else:
                if '-' in str(id):
                    if len(id) == 3:
                        begin, end = id[0], id[2]
                        begin = int(begin)
                        end = int(end)
                        n = 0
                        for x in range(begin, end+1):
                            old2new[x] = pos + n
                            n += 1
                        new_id = (old2new[begin], '-', old2new[end])
                        old2new[id] = new_id
                        pos += end - begin + 1
                    else:
                        raise
                elif '.' in str(id):
                    if len(id) == 3:
                        begin, idx = id[0], id[2]
                        begin = int(begin)
                        idx = int(idx)
                        if begin not in old2new:
                            old2new[begin] = pos
                            pos += 1
                            #raise
                        #old2new[begin] = pos
                        #pos += 1
                        new_id = (old2new[begin], '.', idx)
                        old2new[id] = new_id
                    else:
                        raise

        sent[i]['id'] = old2new[id]

    for tok in sent:
        if 'head' not in tok or tok['head'] is None or tok['head'] == '_':
            continue
        if tok['head'] != 0 and tok['head'] not in old2new:
            raise
        if tok['head'] != 0:
            tok['head'] = old2new[tok['head']]

    pass


def split_sentence(sent, max):
    rv = []
    parts = split_sentence_impl(sent)
    for s in parts:
        if len(s) > max:
            rv.extend(split_sentence(s, max))
        else:
            rv.append(s)
    return rv


def guess_max_sentence_len(tb, out_ratio):
    lengths = []
    for sent in tb:
        lengths.append(len(sent))

    lengths.sort(reverse=True)

    num_outliers = int(len(lengths) * out_ratio)
    if num_outliers < 1:
        num_outliers = 2

    outliers = lengths[:num_outliers]
    max_len = outliers[-1]

    return max_len


def is_sentence_supported(sent):
    for t in sent:
        if isinstance(t['id'], tuple):
            return False
        if isinstance(t['head'], tuple):
            return False
        if t['head'] is None:
            print(str(sent.serialize()))
        i = int(t['head'])
    return True


def augment(tb, max_len):
    new = []

    for sent in tb:
        if not is_sentence_supported(sent):
            print('Not supported:\n%s\n' % str(sent))
            continue
        num = len(sent)
        if num > max_len:
            parts = split_sentence(sent, max_len)
            for s in parts:
                new.append(s)
        else:
            new.append(sent)

    return new


def main():
    if len(sys.argv) <= 1:
        return

    tb = load_conll_file(sys.argv[1])

    new_tb = augment(tb, guess_max_sentence_len(tb, 0.02))

    save_tb(new_tb)


if __name__ == '__main__':
    main()
