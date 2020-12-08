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


def generate_output_line(corpus_name, task_name, stat, curr):
    l = '%36s  LR %.6f  LOSS %9.6f  ACC %s  DEVACC %s' \
        % (
            corpus_name + '/' + task_name,
            curr['param'][task_name]['lr'],
            stat['loss'],
            ('%.6f' % stat['acc']),
            ('%.6f' % stat['devacc'])
        )

    if 'precision' in stat:
        l += '  PR %.6f' % (stat['precision'])
    if 'recall' in stat:
        l += '  RC %.6f' % (stat['recall'])
    if 'f1' in stat:
        l += '  F1 %.6f' % (stat['f1'])

    return l


def generate_output(epoch, tasks, treebank, curr):

    s = 'EPOCH %6d\t%s\n' % (epoch, treebank)

    all_task_names = list(tasks.keys())
    all_task_names.sort()
    for task_name in all_task_names:
        if tasks[task_name]['type'] == 'seqtag':
            s += generate_output_line('', task_name, tasks[task_name], curr) + '\n'
        elif tasks[task_name]['type'] == 'depparse':
            s += generate_output_line('', task_name + '/arcs', tasks[task_name]['arcs'], curr) + '\n'
            s += generate_output_line('', task_name + '/tags', tasks[task_name]['tags'], curr) + '\n'

    return s