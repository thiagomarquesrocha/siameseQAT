from __future__ import print_function, division
import numpy as np
import pandas as pd
import os
import random
import json
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import defaultdict

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
import matplotlib.pyplot as plt

from keras.layers import Conv1D, Input, Add, Activation, Dropout, Embedding, MaxPooling1D, GlobalMaxPool1D, Flatten, Dense, Concatenate, BatchNormalization
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import optimizers
import _pickle as pickle

from multiprocessing import Pool
import multiprocessing
import os

DIR = ''

def read_pairs(df):
  bug_pairs = []
  bug_ids = set()
  for row in df.iterrows():
    duplicates = row[1]['Duplicate']
    bug1 = row[1]['Issue_id']
    duplicates = [] if (type(duplicates) == float) else np.array(duplicates.split(';'), dtype=np.float)
    if len(duplicates) == 0: # No duplicate
      bug_ids.add(int(bug1))
    else: # duplicate
      bug_ids.add(int(bug1))
      for bug2 in duplicates:
        bug_pairs.append((int(bug1), int(bug2)))
        bug_ids.add(int(bug2))
  with open(os.path.join(DIR, 'bug_pairs.txt'), 'w') as f:
    for pair in bug_pairs:
      f.write("%d %d\n" % pair)
  bug_ids = sorted(bug_ids)
  with open(os.path.join(DIR, 'bug_ids.txt'), 'w') as f:
    for bug_id in bug_ids:
      f.write("%d\n" % bug_id)
  return bug_pairs, bug_ids

def split_train_test(bug_pairs, VALIDATION_SPLIT):
  random.shuffle(bug_pairs)
  split_idx = int(len(bug_pairs) * VALIDATION_SPLIT)
  with open(os.path.join(DIR, 'train.txt'), 'w') as f:
    for pair in bug_pairs[:split_idx]:
      f.write("%d %d\n" % pair)
  test_data = {}
  for pair in bug_pairs[split_idx:]:
    bug1 = int(pair[0])
    bug2 = int(pair[1])
    if bug1 not in test_data:
      test_data[bug1] = set()
    test_data[bug1].add(bug2)
  with open(os.path.join(DIR, 'test.txt'), 'w') as f:
    for bug in test_data.keys():
      f.write("{} {}\n".format(bug, ' '.join([str(x) for x in test_data[bug]])))
  print('Train and test created')

def func_name_tokenize(text):
  s = []
  for i, c in enumerate(text):
    if c.isupper() and i > 0 and text[i-1].islower():
      s.append(' ')
    s.append(c)
  return ''.join(s).strip()

def normalize_text(text):
  try:
    tokens = re.compile(r'[\W_]+', re.UNICODE).split(text)
    text = ' '.join([func_name_tokenize(token) for token in tokens])
    text = re.sub(r'\d+((\s\d+)+)?', 'number', text)
  except:
    return 'description'
  return ' '.join([word.lower() for word in nltk.word_tokenize(text)])

def normalized_data(bug_ids, df):
  print("Normalizing text...")
  products = set()
  bug_severities = set()
  priorities = set()
  versions = set()
  components = set()
  bug_statuses = set()
  text = []
  normalized_bugs = open(os.path.join(DIR, 'normalized_bugs.json'), 'w')
  with tqdm(total=df.shape[0]) as loop:
    for row in df.iterrows():
        
        bug = row[1]
        
        bug['description'] = normalize_text(bug['description'])
        if 'title' in bug:
            bug['title'] = normalize_text(bug['title'])
        else:
            bug['title'] = ''
        
        normalized_bugs.write('{}\n'.format(bug.to_json()))

        text.append(bug['description'])
        text.append(bug['title'])
        loop.update(1)
  return text

def build_vocabulary(train_text, MAX_NB_WORDS):
  word_freq = build_freq_dict(train_text)
  print('word vocabulary')
  word_vocab = save_vocab(word_freq, MAX_NB_WORDS, 'word_vocab.pkl')
  return word_vocab

def build_freq_dict(train_text):
  print('building frequency dictionaries')
  word_freq = defaultdict(int)
  for text in tqdm(train_text):
    for word in text.split():
      word_freq[word] += 1
  return word_freq

def save_vocab(freq_dict, vocab_size, filename):
  top_tokens = sorted(freq_dict.items(), key=lambda x: -x[1])[:vocab_size - 2]
  print('most common token is %s which appears %d times' % (top_tokens[0][0], top_tokens[0][1]))
  print('less common token is %s which appears %d times' % (top_tokens[-1][0], top_tokens[-1][1]))
  vocab = {}
  i = 2  # 0-index is for padding, 1-index is for UNKNOWN
  for j in range(len(top_tokens)):
    vocab[top_tokens[j][0]] = i
    i += 1
  with open(os.path.join(DIR, filename), 'wb') as f:
    pickle.dump(vocab, f)
  return vocab

def load_vocab(filename):
    with open(os.path.join(DIR, filename), 'rb') as f:
        return pickle.load(f)

def dump_bugs(word_vocab, total):
    bug_dir = os.path.join(DIR, 'bugs')
    if not os.path.exists(bug_dir):
        os.mkdir(bug_dir)
    bugs = []
    cont = 1
    print("Reading the normalized_bugs.json ...")
    with open(os.path.join(DIR, 'normalized_bugs.json'), 'r') as f:
        #loop = tqdm(f)
        with tqdm(total=total) as loop:
            for line in f:
                #loop.set_description('Data dumping {}/{}'.format(cont, total))
                loop.update(1)
                bugs.append(json.loads(line))
                cont += 1

    return bugs

def dump_vocabulary(bugs, word_vocab, bug_dir):
    UNK = 1
    cont=0
    total = len(bugs)
    print("Starting the dump ...")
    for bug in tqdm(bugs):
        #bug = json.loads(line)
        #print(bug)
        cont+=1
        bug['description_word'] = [word_vocab.get(w, UNK) for w in bug['description'].split()]
        if len(bug['title']) == 0:
            bug['title'] = bug['description'][:10]
        bug['title_word'] = [word_vocab.get(w, UNK) for w in bug['title'].split()]
        #bug.pop('description')
        #bug.pop('title')
        with open(os.path.join(bug_dir, str(bug['issue_id']) + '.pkl'), 'wb') as f:
            pickle.dump(bug, f)

def processing_dump(bugs, word_vocab):
    #clear_output()
    cpu = os.cpu_count() - 1
    pool = Pool(processes=cpu) # start 4 worker processes
    bug_dir = os.path.join(DIR, 'bugs')
    print("Starting the slice ...")
    works = []
    n = len(bugs) // cpu
    n = 1 if n == 0 else n
    sliced = []
    pos_end = n
    end = len(bugs)
    for i in range(cpu):
        pos_end = end if pos_end>=end else pos_end
        pos_end = end if (i+1) == cpu and pos_end < end else pos_end
        sliced.append(bugs[i*n:pos_end])
        pos_end += n

    print("Slicing done!")
    for s in sliced:
        if len(s) > 0:
            works.append(pool.apply_async(dump_vocabulary, (s, word_vocab, bug_dir, )))
            #dump_vocabulary(s, bug_dir)

    print("Executing the works...")
    res = [w.get() for w in works]

    # dump_vocabulary(bugs, word_vocab, bug_dir)

    print("All done!")

def main():
    MAX_NB_WORDS = 20000
    VALIDATION_SPLIT = 0.8
    
    # Train
    df_train = pd.read_csv('mozilla_firefox.csv')
    df_train.columns = ['issue_id', 'priority', 'component', 'duplicated_issue', 'title', 'description', 'status', 'resolution', 'version', 'creation_ts', 'delta_ts']

    ### Pairs
    df_train_pair = pd.read_csv('train_mozilla_firefox.csv')

    bug_pairs, bug_ids = read_pairs(df_train_pair)
    print("Number of bugs: {}".format(len(bug_ids)))
    print("Number of pairs: {}".format(len(bug_pairs)))

    # Split into train/test
    split_train_test(bug_pairs, VALIDATION_SPLIT)
    # Normalize the text
    text = normalized_data(bug_ids, df_train)
    # Build the vocab
    word_vocab = build_vocabulary(text, MAX_NB_WORDS)
    
    # Dump the preprocessed bugs
    num_lines =  len(open(os.path.join(DIR, 'normalized_bugs.json'), 'r').read().splitlines()) * 2
    total = num_lines // 2
    bugs = dump_bugs(word_vocab, total)
    processing_dump(bugs, word_vocab)
    print("Saved!")


main()