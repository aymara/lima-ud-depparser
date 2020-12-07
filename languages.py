#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os
import argparse

from data import ud_guess_base_name, ud_treebank_path


languages = [
  {
    'lang':     'Afrikaans',
    'wiki':     'af',
    'ft':       'af', 
    'iso639-1': 'af',
    'iso639-2': 'afr',
    'iso639-3': 'afr',
    'corpora': [
      { 'name': 'AfriBooms', 'train': True }
    ]
  },

  {
    'lang':     'Akkadian',
    'corpora': [
      { 'name': 'PISANDUB',  'train': False }, # no train set available in UD 2.4
    ]
  },

  {
    'lang':     'Albanian', # new in UD 2.6
    'wiki':     'sq',
    'ft':       'sq',
    #'iso639-1': 'af',
    #'iso639-2': 'afr',
    'iso639-3': 'sqi',
    'corpora': [
      { 'name': 'TSA',  'train': False }, # no train set available in UD 2.4
      { 'name': 'Synt', 'train': True,
        'hp': {
            'batchSize': 10,
            'embdDim': 300,
            'lstmUnits': 100,
            'lstmLayers': 3,
            'wordEmbdL2': 0.5,
            'charEmbdL2': 0.5,
            'dpSpecificLstm': {
                'layers': [50, 50]
            }
        }
      }
    ]
  },

  {
    'lang':     'Amharic',
    'ft':       'am',
    'wiki':     'am',
    'iso639-1': 'am',
    'iso639-2': 'amh',
    'iso639-3': 'amh',
    'corpora': [
      { 'name': 'ATT', 'train': False },  # no train set available in UD 2.4
    ]
  },

  { 
    'lang':     'Ancient_Greek', # no fastText, several wiki resources exist
    'iso639-2': 'grc',
    'iso639-3': 'grc',
    'corpora': [
      { 'name': 'Perseus', 'train': False },
      { 'name': 'PROIEL',  'train': False }
    ]
  },

  {
    'lang':     'Arabic',
    'ft':       'ar', 
    'wiki':     'ar',
    'iso639-1': 'ar',
    'iso639-2': 'ara',
    'iso639-3': 'ara',
    'corpora': [
      { 'name': 'NYUAD', 'train': False }, # no text in UD 2.4
      { 'name': 'PADT',  'train': True, 'big': True,
        'hp': {
          #'clipping': 1.0
        }
      },
      { 'name': 'PUD',   'train': False } # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Armenian',
    'ft':       'hy',
    'wiki':     '',
    'iso639-1': 'hy',
    'iso639-2': 'hye',
    'iso639-3': 'hye',
    'corpora': [
      { 'name': 'ArmTDP', 'train': True }
    ]
  },

  {
    'lang':     'Assyrian',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name':'AS', 'train': False }
    ]
  },

  {
    'lang':     'Bambara',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name':'CRB', 'train': False }
    ]
  },

  {
    'lang':     'Basque',
    'ft':       'eu',
    'wiki':     'eu',
    'iso639-1': 'eu',
    'iso639-2': 'eus',
    'iso639-3': 'eus',
    'corpora': [
      { 'name': 'BDT', 'train': True }
    ]
  },

  {
    'lang':     'Belarusian',
    'ft':       'be',
    'wiki':     'be',
    'iso639-1': 'be',
    'iso639-2': 'bel',
    'iso639-3': 'bel',
    'corpora': [
      { 'name': 'HSE', 'train': True }
    ]
  },

  {
    'lang':     'Breton',
    'ft':       'br',
    'wiki':     'br',
    'iso639-1': 'br',
    'iso639-2': 'bre',
    'iso639-3': 'bre',
    'corpora': [
      { 'name': 'KEB', 'train': False,
        'hp': {
          'batchSize': 80,
          'embdDim': 300,
          'lstmUnits': 100,
          'lstmLayers': 3,
          'wordEmbdL2': 0.5,
          'charEmbdL2': 0.5,
          'dpSpecificLstm': {
            'layers': [50, 50]
          }
        }
      } # no train set in UD 2.4
    ]
  },

  {
    'lang':     'Bulgarian',
    'ft':       'bg',
    'wiki':     'bg',
    'iso639-1': 'bg',
    'iso639-2': 'bul',
    'iso639-3': 'bul',
    'corpora': [
      { 'name': 'BTB', 'train': True }
    ]
  },

  {
    'lang':     'Buryat', # Russia Buryat with Cyrillic script
    'wiki':     'bxr',
    'iso639-2': 'bua',
    'iso639-3': 'bxr',
    'corpora': [
      { 'name': 'BDT', 'train': False, } # train set is too small and no dev set in UD 2.4
    ]
  },

  {
    'lang':     'Cantonese',
    'wiki':     'zh-yue',
    'iso639-3': 'yue',
    'corpora': [
      { 'name':'HK', 'train': False } # no train set in UD 2.4
                                      # traditional characters
    ]
  },

  {
    'lang':     'Catalan',
    'ft':       'ca',
    'wiki':     'ca',
    'iso639-1': 'ca',
    'iso639-2': 'cat',
    'iso639-3': 'cat',
    'corpora': [
      { 'name': 'AnCora', 'train': True, 'big': True }
    ]
  },

  {
    'lang':     'Chinese',
    'ft':       'zh',
    'wiki':     'zh',
    'iso639-1': 'zh',
    'iso639-2': 'zho',
    'iso639-3': 'zho',
    'corpora': [
      { 'name': 'CFL', 'train': False }, # no train set in UD 2.4
      { 'name': 'GSD', 'train': True  }, # traditional Chinese Universal Dependencies Treebank annotated and converted by Google
      { 'name': 'HK',  'train': False }, # no train set in UD 2.4
                                         # parallel with Cantonese-HK
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.4
    ]
  },

  {
    'lang': 'Chinese',
    'ft': 'zh',
    'wiki': 'zh',
    'iso639-1': 'zh',
    'iso639-2': 'zho',
    'iso639-3': 'zho-simp',
    'corpora': [
      {'name': 'GSDSimp', 'train': True},  # GSD converted to simplified Chinese
    ]
  },

  {
    'lang':     'Classical_Chinese', # Literary Chinese
                                     # no fastText
    'ft':       'train',
    'wiki':     'lzh',
    'iso639-1': 'zh',
    'iso639-3': 'lzh',
    'corpora': [
      { 'name':'Kyoto',
        'hp': {
          'batchSize': 80,
          'embdDim': 300,
          'lstmUnits': 100,
          'lstmLayers': 3,
          'wordEmbdL2': 0.5,
          'charEmbdL2': 0.5,
          'dpSpecificLstm': {
            'layers': [50, 50]
          }
        }
      }
    ]
  },

  {
    'lang':     'Coptic', # no fastText
    'ft':       'train',
    #'wiki':     '', # incubator only
    'iso639-2': 'cop',
    'iso639-3': 'cop',
    'corpora': [
      { 'name':'Scriptorium', 'train': True } 
    ]
  },

  { 'lang':     'Croatian',
    'ft':       'hr',
    'wiki':     'hr',
    'iso639-1': 'hr',
    'iso639-2': 'hrv',
    'iso639-3': 'hrv',
    'corpora': [
      { 'name': 'SET', 'train': True }
    ]
  },

  {
    'lang':     'Czech',
    'ft':       'cs',
    'wiki':     'cs',
    'iso639-1': 'cs',
    'iso639-2': 'ces',
    'iso639-3': 'ces',
    'corpora': [
      { 'name': 'PDT',     'train': True, 'big': True }, # main corpus
      { 'name': 'CAC',     'train': True, 'big': True },
      { 'name': 'CLTT',    'train': False }, # GPU OOM (too many features)
      { 'name': 'FicTree', 'train': True },
      { 'name': 'PUD',     'train': False } # no train set in UD 2.4
    ]
  },

  {
    'lang':     'Danish',
    'ft':       'da',
    'wiki':     'da',
    'iso639-1': 'da',
    'iso639-2': 'dan',
    'iso639-3': 'dan',
    'corpora': [
      { 'name': 'DDT', 'train': True }
    ]
  },

  {
    'lang':     'Dutch',
    'ft':       'nl',
    'wiki':     'nl',
    'iso639-1': 'nl',
    'iso639-2': 'nld',
    'iso639-3': 'nld',
    'corpora': [
      {'name': 'LassySmall', 'train': True},
      { 'name': 'Alpino',     'train': True }
    ]
  },

  {
    'lang':     'English',
    'ft':       'en',
    'wiki':     'en',
    'iso639-1': 'en',
    'iso639-2': 'eng',
    'iso639-3': 'eng',
    'corpora': [
      { 'name': 'EWT', 'train': True, 'big': True },
      { 'name': 'ESL', 'train': False },
      { 'name': 'GUM' },
      { 'name': 'LinES' },
      { 'name': 'ParTUT' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
      { 'name': 'Pronouns', 'train': False } # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Erzya',
    'ft':       'myv',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name': 'JR', 'train': False } # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Estonian',
    'ft':       'et',
    'wiki':     '',
    'iso639-1': 'et',
    'iso639-2': 'est',
    'iso639-3': 'est',
    'corpora': [
      { 'name': 'EDT', 'train': True, 'big': True },
      { 'name': 'EWT', 'train': False } # no dev set in UD 2.5 
    ]
  },

  {
    'lang':     'Faroese', # no fastText
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name':'OFT', 'train': False } # no train set in UD 2.4
    ]
  },

  {
    'lang':     'Finnish',
    'ft':       'fi',
    'wiki':     'fi',
    'iso639-1': 'fi',
    'iso639-2': 'fin',
    'iso639-3': 'fin',
    'corpora': [
      { 'name': 'TDT', 'train': True, 'big': True },
      { 'name': 'FTB' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5

    ]
  },

  {
    'lang':     'French',
    'ft':       'fr',
    'wiki':     'fr',
    'iso639-1': 'fr',
    'iso639-2': 'fre',
    'iso639-3': 'fra',
    'corpora': [
      { 'name': 'Sequoia' },
      { 'name': 'FQB', 'train': False }, # no train set in UD 2.5
      { 'name': 'FTB', 'train': False },
      { 'name': 'GSD', 'train': True, 'big': True },
      { 'name': 'ParTUT' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
      { 'name': 'Spoken' }
    ]
  },

  {
    'lang':     'Galician',
    'ft':       'gl',
    'wiki':     '',
    'iso639-1': 'gl',
    'iso639-2': 'glg',
    'iso639-3': 'glg',
    'corpora': [
      { 'name': 'CTG',     'train': True },
      { 'name': 'TreeGal', 'train': False } # no dev set in UD 2.5
    ]
  },

  {
    'lang':     'German',
    'ft':       'de',
    'wiki':     'de',
    'iso639-1': 'de',
    'iso639-2': 'deu',
    'iso639-3': 'deu',
    'corpora': [
      { 'name': 'HDT', 'train': True, 'big': True },
      { 'name': 'GSD' },
      { 'name': 'LIT', 'train': False }, # no train set in UD 2.5
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Gothic', # no fastText
    'ft':       'train',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': 'got',
    'iso639-3': 'got',
    'corpora': [
      { 'name':'PROIEL', 'train': False,
        'hp': {
          'learningRate': {
            'depparse': {
              #'arcs': 0.0001
            }
          },
        }
      } 
    ]
  },

  {
    'lang':     'Greek',
    'ft':       'el',
    'wiki':     '',
    'iso639-1': 'el',
    'iso639-2': 'ell',
    'iso639-3': 'ell',
    'corpora': [
      { 'name': 'GDT', 'train': True }
    ]
  },

  {
    'lang':     'Hebrew',
    'ft':       'he',
    'wiki':     '',
    'iso639-1': 'he',
    'iso639-2': 'heb',
    'iso639-3': 'heb',
    'corpora': [
      { 'name': 'HTB', 'train': True }
    ]
  },

  { 'lang':     'Hindi_English', # Hinglish
                                 # no fastText
    'ft':       'train',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name':'HIENCS', 'train': False } # no text available in UD 2.4 (can be downloaded)
    ]
  },

  {
    'lang':     'Hindi',
    'ft':       'hi',
    'wiki':     '',
    'iso639-1': 'hi',
    'iso639-2': 'hin',
    'iso639-3': 'hin',
    'corpora': [
      { 'name': 'HDTB', 'train': True },
      { 'name': 'PUD', 'train': False } # not train set available for UD 2.4
    ]
  },

  {
    'lang':     'Hungarian',
    'ft':       'hu',
    'wiki':     '',
    'iso639-1': 'hu',
    'iso639-2': 'hun',
    'iso639-3': 'hun',
    'corpora': [
      { 'name': 'Szeged' }
    ]
  },

  {
    'lang':     'Indonesian',
    'ft':       'id',
    'wiki':     '',
    'iso639-1': 'id',
    'iso639-2': 'ind',
    'iso639-3': 'ind',
    'corpora': [
      { 'name': 'GSD' },
      { 'name': 'PUD', 'train': False } # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Irish',
    'ft':       'ga',
    'wiki':     '',
    'iso639-1': 'ga',
    'iso639-2': 'gle',
    'iso639-3': 'gle',
    'corpora': [
      { 'name': 'IDT' }
    ]
  },

  {
    'lang':     'Italian',
    'ft':       'it',
    'wiki':     '',
    'iso639-1': 'it',
    'iso639-2': 'ita',
    'iso639-3': 'ita',
    'corpora': [
      { 'name': 'ISDT' },
      { 'name': 'ParTUT' },
      { 'name': 'PoSTWITA' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
      { 'name': 'VIT' },
      { 'name': 'TWITTIRO' }
    ]
  },

  {
    'lang':     'Japanese',
    'ft':       'ja',
    'wiki':     '',
    'iso639-1': 'ja',
    'iso639-2': 'jpn',
    'iso639-3': 'jpn',
    'corpora': [
      { 'name': 'BCCWJ', 'train': False },
      { 'name': 'GSD' },
      { 'name': 'Modern', 'train': False }, # no train set in UD 2.5
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
    ]
  },

  { 
    'lang':     'Karelian', # no fastText
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name':'KKPP', 'train': False } # no dev set in UD 2.5
    ]
  },

  {
    'lang':     'Kazakh',
    'ft':       'kk',
    'wiki':     'kk',
    'iso639-1': 'kk',
    'iso639-2': 'kaz',
    'iso639-3': 'kaz',
    'corpora': [
      { 'name': 'KTB',
        'hp': {
            'batchSize': 10,
            'embdDim': 300,
            'lstmUnits': 100,
            'lstmLayers': 3,
            'wordEmbdL2': 0.5,
            'charEmbdL2': 0.5,
            'dpSpecificLstm': {
                'layers': [50, 50]
            }
        }
      }, # no dev set in UD 2.5
    ]
  },

  {
    'lang':     'Komi_Zyrian', # no fastText
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name':'IKDP',    'train': False }, # no train set in UD 2.4
      { 'name':'Lattice', 'train': False }, # no train set in UD 2.4
    ]
  },

  {
    'lang':     'Korean',
    'ft':       'ko',
    'wiki':     '',
    'iso639-1': 'ko',
    'iso639-2': 'kor',
    'iso639-3': 'kor',
    'corpora': [
      { 'name': 'Kaist' },
      { 'name': 'GSD' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Kurmanji',
    'ft':       'ku',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': 'kmr',
    'corpora': [
      { 'name': 'MG',
        'hp': {
          'batchSize': 10,
          'embdDim': 300,
          'lstmUnits': 100,
          'lstmLayers': 3,
          'wordEmbdL2': 0.5,
          'charEmbdL2': 0.5,
          'dpSpecificLstm': {
            'layers': [50, 50]
          }
        }
      }, # no dev set in UD 2.5
    ]
  },

  {
    'lang':     'Latin',
    'ft':       'la',
    'wiki':     '',
    'iso639-1': 'la',
    'iso639-2': 'lat',
    'iso639-3': 'lat',
    'corpora': [
      { 'name': 'PROIEL', 'train': True, 'big': True },
      { 'name': 'ITTB' },
      { 'name': 'Perseus', 'train': False } # no dev set in UD 2.5

    ]
  },

  {
    'lang':     'Latvian',
    'ft':       'lv',
    'wiki':     '',
    'iso639-1': 'lv',
    'iso639-2': 'lav',
    'iso639-3': 'lav',
    'corpora': [
      { 'name': 'LVTB', 'train': True, 'big': True }
    ]
  },

  {
    'lang':     'Lithuanian',
    'ft':       'lt',
    'wiki':     '',
    'iso639-1': 'lt',
    'iso639-2': 'lit',
    'iso639-3': 'lit',
    'corpora': [
      { 'name': 'ALKSNIS' },
      { 'name': 'HSE' }
    ]
  },

  {
    'lang':     'Maltese',
    'ft':       'mt',
    'wiki':     '',
    'iso639-1': 'mt',
    'iso639-2': 'mlt',
    'iso639-3': 'mlt',
    'corpora': [
      { 'name': 'MUDT' }
    ]
  },

  {
    'lang':     'Marathi',
    'ft':       'mr',
    'wiki':     '',
    'iso639-1': 'mr',
    'iso639-2': 'mar',
    'iso639-3': 'mar',
    'corpora': [
      { 'name': 'UFAL' },
    ]
  },

  {
    'lang':'Mbya_Guarani',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': 'gun',
    'corpora': [
      { 'name':'Dooley', 'train': False }, # no text available
                                           # no train set in UD 2.4
      { 'name':'Thomas', 'train': False }  # no train set in UD 2.4
    ]
  },

  {
    'lang':     'Naija', # no fastText
    'ft':       'train',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': 'pcm',
    'corpora': [
      { 'name':'NSC', 'use_test_set': True }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'North_Sami', # no fastText
    'ft':       'train',
    'wiki':     '',
    'iso639-1': 'se',
    'iso639-2': 'sme',
    'iso639-3': 'sme',
    'corpora': [
      { 'name':'Giella', #'use_test_set': True,
        'hp': {
          'embdDim': 100,
          #'charEmbdDim': 16,
          #'charLstmUnits': 32,
          'lstmUnits': 100,
          'lstmLayers': 3,
          'wordEmbdL2': 0.1,
          'charEmbdL2': 0.1,
          'dpSpecificLstm': {
            'layers': [50, 50]
          }
        }
      }, # no dev set in UD 2.5
    ]
  },

  {
    'lang':     'Norwegian',
    'ft':       'no',
    'wiki':     '',
    'iso639-1': 'nb',
    'iso639-2': 'nob',
    'iso639-3': 'nob',
    'corpora': [
      { 'name': 'Bokmaal' }
    ]
  },

  {
    'lang':     'Norwegian',
    'ft':       'nn',
    'wiki':     '',
    'iso639-1': 'nn',
    'iso639-2': 'nno',
    'iso639-3': 'nno',
    'corpora': [
      { 'name': 'Nynorsk' }
    ]
  },

  {
    'lang':     'Norwegian',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name':'NynorskLIA', 'train': False }, # no fastText
    ]
  },

  {
    'lang':     'Old_Church_Slavonic', # no fastText
    'ft':       'train',
    'wiki':     '',
    'iso639-1': 'cu',
    'iso639-2': 'chu',
    'iso639-3': 'chu',
    'corpora': [
      { 'name':'PROIEL', 'train': True },
    ]
  },

  {
    'lang':     'Old_French', # no fastText
    'ft':       'train',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': 'fro',
    'iso639-3': 'fro',
    'corpora': [
      { 'name':'SRCMF', 'train': True },
    ]
  },

  {
    'lang':     'Old_Russian', # no fastText
    'ft':       'train',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': 'orv', # ?
    'corpora': [
      { 'name':'RNC',   'train': False }, # no train set in UD 2.5
      { 'name':'TOROT', 'train': True },
    ]
  },

  { 'lang':     'Persian',
    'ft':       'fa',
    'wiki':     'fa',
    'iso639-1': 'fa',
    'iso639-2': 'fas',
    'iso639-3': 'fas',
    'corpora': [
      { 'name':'Seraji' }
    ]
  },

  {
    'lang':     'Polish',
    'ft':       'pl',
    'wiki':     '',
    'iso639-1': 'pl',
    'iso639-2': 'pol',
    'iso639-3': 'pol',
    'corpora': [
      { 'name': 'PDB', 'train': True, 'big': True },
      { 'name': 'LFG' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Portuguese',
    'ft':       'pt',
    'wiki':     '',
    'iso639-1': 'pt',
    'iso639-2': 'por',
    'iso639-3': 'por',
    'corpora': [
      { 'name': 'GSD' },
      { 'name': 'Bosque' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Romanian',
    'ft':       'ro',
    'wiki':     '',
    'iso639-1': 'ro',
    'iso639-2': 'ron',
    'iso639-3': 'ron',
    'corpora': [
      { 'name': 'Nonstandard' },
      { 'name': 'RRT' },
      { 'name': 'SiMoNERo', 'train': False} # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Russian',
    'ft':       'ru',
    'wiki':     '',
    'iso639-1': 'ru',
    'iso639-2': 'rus',
    'iso639-3': 'rus',
    'corpora': [
      { 'name': 'SynTagRus', 'train': True, 'big': True },
      { 'name': 'GSD', 'train': False },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
      { 'name': 'Taiga', 'train': False }
    ]
  },

  {
    'lang':     'Sanskrit',
    'ft':       'sa',
    'wiki':     '',
    'iso639-1': 'sa',
    'iso639-2': 'san',
    'iso639-3': 'san',
    'corpora': [
      { 'name': 'UFAL', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Serbian',
    'ft':       'sr',
    'wiki':     '',
    'iso639-1': 'sr',
    'iso639-2': 'srp',
    'iso639-3': 'srp',
    'corpora': [
      { 'name': 'SET' }
    ]
  },

  {
    'lang':     'Slovak',
    'ft':       'sk',
    'wiki':     '',
    'iso639-1': 'sk',
    'iso639-2': 'slk',
    'iso639-3': 'slk',
    'corpora': [
      { 'name': 'SNK' }
    ]
  },

  {
    'lang':     'Slovenian',
    'ft':       'sl',
    'wiki':     '',
    'iso639-1': 'sl',
    'iso639-2': 'slv',
    'iso639-3': 'slv',
    'corpora': [
      { 'name': 'SSJ' },
      { 'name': 'SST', 'train': False }, # no dev set in UD 2.5
    ]
  },

  {
    'lang':     'Spanish',
    'ft':       'es',
    'wiki':     'es',
    'iso639-1': 'es',
    'iso639-2': 'spa',
    'iso639-3': 'spa',
    'corpora': [
      { 'name': 'AnCora', 'train': True, 'big': True },
      { 'name': 'GSD' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Swedish',
    'ft':       'sv',
    'wiki':     '',
    'iso639-1': 'sv',
    'iso639-2': 'swe',
    'iso639-3': 'swe',
    'corpora': [
      { 'name': 'Talbanken' },
      { 'name': 'LinES' },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':'Swedish_Sign_Language', # no fastText
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      { 'name':'SSLC', 'train': False } # train set is too small in UD 2.5
    ]
  },

  {
    'lang':     'Tagalog',
    'ft':       'tl',
    'wiki':     '',
    'iso639-1': 'tl',
    'iso639-2': 'tgl',
    'iso639-3': 'tgl',
    'corpora': [
      { 'name': 'TRG', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Tamil',
    'ft':       'ta',
    'wiki':     '',
    'iso639-1': 'ta',
    'iso639-2': 'tam',
    'iso639-3': 'tam',
    'corpora': [
      { 'name': 'TTB' }
    ]
  },

  {
    'lang':     'Telugu',
    'ft':       'te',
    'wiki':     '',
    'iso639-1': 'te',
    'iso639-2': 'tel',
    'iso639-3': 'tel',
    'corpora': [
      { 'name': 'MTG' }
    ]
  },

  {
    'lang':     'Thai',
    'ft':       'th',
    'wiki':     '',
    'iso639-1': 'th',
    'iso639-2': 'tha',
    'iso639-3': 'tha',
    'corpora': [
      { 'name': 'PUD',
        'hp': {
          'batchSize': 80,
          'embdDim': 300,
          'lstmUnits': 100,
          'lstmLayers': 3,
          'wordEmbdL2': 0.5,
          'charEmbdL2': 0.5,
          'dpSpecificLstm': {
            'layers': [50, 50]
          }
        }
      }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Turkish',
    'ft':       'tr',
    'wiki':     '',
    'iso639-1': 'tr',
    'iso639-2': 'tur',
    'iso639-3': 'tur',
    'corpora': [
      { 'name': 'GB', 'train': False }, # no train set in UD 2.5
      { 'name': 'IMST',
        'hp': {
          'batchSize': 80,
          'embdDim': 300,
          'lstmUnits': 100,
          'lstmLayers': 3,
          'wordEmbdL2': 0.5,
          'charEmbdL2': 0.5,
          'dpSpecificLstm': {
            'layers': [50, 50]
          }
        }
      },
      { 'name': 'PUD', 'train': False }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Ukrainian',
    'ft':       'uk',
    'wiki':     '',
    'iso639-1': 'uk',
    'iso639-2': 'ukr',
    'iso639-3': 'ukr',
    'corpora': [
      { 'name': 'IU' }
    ]
  },

  {
    'lang':     'Upper_Sorbian',
    'ft':       'hsb',
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': 'hsb',
    'iso639-3': 'hsb',
    'corpora': [
      { 'name': 'UFAL', 'train': False }, # no dev set in UD 2.5
    ]
  },

  {
    'lang':     'Urdu',
    'ft':       'ur',
    'wiki':     '',
    'iso639-1': 'ur',
    'iso639-2': 'urd',
    'iso639-3': 'urd',
    'corpora': [
      { 'name': 'UDTB' }
    ]
  },

  {
    'lang':     'Uyghur',
    'ft':       'ug',
    'wiki':     '',
    'iso639-1': 'ug',
    'iso639-2': 'uig',
    'iso639-3': 'uig',
    'corpora': [
      { 'name': 'UDT' }
    ]
  },

  {
    'lang':     'Vietnamese',
    'ft':       'vi',
    'wiki':     '',
    'iso639-1': 'vi',
    'iso639-2': 'vie',
    'iso639-3': 'vie',
    'corpora': [
      { 'name': 'VTB' },
    ]
  },

  {
    'lang':     'Warlpiri', # no fastText
    'wiki':     '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': 'wbp',
    'corpora': [
      { 'name':'UFAL', 'train': False }, # no train set available in UD 2.4
    ]
  },

  {
    'lang':     'Welsh',
    'ft':       'cy',
    'wiki':     '',
    'iso639-1': 'cy',
    'iso639-2': 'cym',
    'iso639-3': 'cym',
    'corpora': [
      { 'name': 'CCG', 'use_test_set': True }, # no train set in UD 2.5
    ]
  },

  {
    'lang':     'Wolof', # no fastText
    'ft':       'train',
    'wiki':     'wo',
    'iso639-1': 'wo',
    'iso639-2': 'wol',
    'iso639-3': 'wol',
    'corpora': [
      { 'name':'WTB', 'train': True }
    ]
  },

  {
    'lang':     'Yoruba',
    'ft':       'yo',
    'wiki':     '',
    'iso639-1': 'yo',
    'iso639-2': 'yor',
    'iso639-3': 'yor',
    'corpora': [
      { 'name': 'YTB', 'train': False }, # no train set in UD 2.5
    ]
  },

  # UD 2.5

  # {
  #   'lang': '',
  #   'ft': '',
  #   'wiki': '',
  #   'iso639-1': '',
  #   'iso639-2': '',
  #   'iso639-3': '',
  #   'corpora': [
  #     {'name': ''}
  #   ]
  # },

  {
    'lang': 'Bhojpuri', # no fastText
    'wiki': '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      {'name': 'BHTB', 'train': False } # no train set in UD 2.5
    ]
  },

  {
    'lang': 'Komi_Permyak', # no fastText
    'wiki': '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      {'name': 'UH', 'train': False } # no train set in UD 2.5
    ]
  },

  {
    'lang': 'Livvi', # no fastText
    'wiki': '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      {'name': 'KKPP', 'train': False }
    ]
  },

  {
    'lang': 'Moksha', # no fastText
    'wiki': '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      {'name': 'JR', 'train': False } # no train set in UD 2.5
    ]
  },

  {
    'lang': 'Scottish_Gaelic',
    'ft': 'gd',
    'wiki': '',
    'iso639-1': 'gd',
    'iso639-2': 'gla',
    'iso639-3': 'gla',
    'corpora': [
      {'name': 'ARCOSG', 'train': True }
    ]
  },

  {
    'lang': 'Swiss_German', # no fastText
    'wiki': '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      {'name': 'UZH', 'train': False } # no train set in UD 2.5
    ]
  },

  {
    'lang': 'Skolt_Sami',
    'ft': '',
    'wiki': '',
    'iso639-1': '',
    'iso639-2': '',
    'iso639-3': '',
    'corpora': [
      {'name': 'Giellagas', 'train': False } # no train set in UD 2.5
    ]
  },

]


def get_trainable_corpora(lang):
  if 'ft' not in lang or lang['ft'] is None or len(lang['ft']) == 0 :
    return []

  rv = []
  for corpus in lang['corpora']:
    if 'train' in corpus and not corpus['train']:
      continue
    rv.append(corpus)

  return rv


def get_fields(lang, corpus, fields):
  rv = []
  if not isinstance(fields, list):
    fields = [ fields ]

  for field in fields:
    if '.' in field:
      parts = field.split('.')
      if parts[0] == 'corpus' and parts[1] in corpus:
        if parts[1] == 'name':
          rv.append('-'.join([lang['lang'], corpus[parts[1]]]))
        else:
          rv.append(corpus[parts[1]])
      else:
        raise
    else:
      if field in lang:
        rv.append(lang[field])
      else:
        rv.append('None')

  return rv


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--fields', type=str, help='List of fields to print')
  parser.add_argument('-m', '--mode', choices=['all', 'trainable', 'nontrainable'], help='Print all corpora or only trainable')
  parser.add_argument('-u', '--ud-path', help='UD path')
  parser.add_argument('-c', '--check', action='store_true', help='Check availability of corpora')
  parser.add_argument('-g', '--get-ft', type=str, help='Get fastText code')
  parser.add_argument('-i', '--get-iso639-3', type=str, help='Get ISO-639-3 code')
  args = parser.parse_args()

  if args.get_ft is not None:
    lang = args.get_ft
    if '-' in lang:
      lang = lang.split('-')[0]
    for l in languages:
      if lang == l['lang']:
        print(l['ft'])
        sys.exit()

  if args.get_iso639_3 is not None:
    lang = args.get_iso639_3
    if '-' in lang:
      lang = lang.split('-')[0]
    for l in languages:
      if lang == l['lang']:
        print(l['iso639-3'])
        sys.exit()

  fields = args.fields.split(',')

  for lang in languages:
    corpora = lang['corpora']

    if args.mode == 'trainable':
      corpora = get_trainable_corpora(lang)
      if len(corpora) == 0:
        continue

    if args.mode == 'nontrainable':
      trainable_corpora = [ c['name'] for c in get_trainable_corpora(lang) ]
      nontrainable_corpora = []
      for c in corpora:
        if c['name'] in trainable_corpora:
          continue
        nontrainable_corpora.append(c)
      corpora = nontrainable_corpora

    for corpus in corpora:
      line = '\t'.join(get_fields(lang, corpus, fields))
      print(line)
      if args.check:
        treebank = get_fields(lang, corpus, 'corpus.name')[0]
        treebank_path = ud_treebank_path(args.ud_path, treebank)
        base_name = ud_guess_base_name(treebank_path, '.conllu')
        if base_name is None:
          sys.stderr.write('ERROR: can\'t guess base name for %s\n' % treebank)
          raise
        train_file_name = os.path.join(treebank_path, base_name) + "-train.conllu"
        if not os.path.isfile(train_file_name):
          sys.stderr.write('ERROR: %s isn\'t found\n' % train_file_name)
          raise
        dev_file_name = os.path.join(treebank_path, base_name) + "-dev.conllu"
        if not os.path.isfile(dev_file_name):
          sys.stderr.write('ERROR: %s isn\'t found\n' % dev_file_name)
          raise
        test_file_name = os.path.join(treebank_path, base_name) + "-test.conllu"
        if not os.path.isfile(test_file_name):
          sys.stderr.write('ERROR: %s isn\'t found\n' % test_file_name)
          raise


def get_language_descr(name):
  for lang in languages:
    if lang['lang'] == name:
      return lang
  return None


def merge_hp(dest, src):
  for k in src:
    if isinstance(src[k], dict):
      if k not in dest:
        dest[k] = {}
      merge_hp(dest[k], src[k])
    else:
      dest[k] = src[k]


def update_hp_for_corpus(hp, treebank):
  language, corpus = treebank.split('-')
  lang_descr = get_language_descr(language)
  if lang_descr is None:
    sys.stderr.write('Error: unknown language \'%s\'\n' % (language))
    raise

  for corpus_descr in lang_descr['corpora']:
    if corpus_descr['name'] == corpus:
      break

  if 'hp' in corpus_descr:
    merge_hp(hp, corpus_descr['hp'])


if __name__ == '__main__':
    main()
