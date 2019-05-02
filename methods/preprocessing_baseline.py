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

import _pickle as pickle

from multiprocessing import Pool
import multiprocessing
import sys
import spacy

class Preprocess:

  ENTITY_ENUM = {
    '': '',
    'PERSON': 'person',
    'NORP': 'nationality',
    'FAC': 'facility',
    'ORG': 'organization',
    'GPE': 'country',
    'LOC': 'location',
    'PRODUCT': 'product',
    'EVENT': 'event',
    'WORK_OF_ART': 'artwork',
    'LANGUAGE': 'language',
    'DATE': 'date',
    'TIME': 'time',
    'PERCENT': 'percent',
    'MONEY': 'money',
    'QUANTITY': 'quantity',
    'ORDINAL': 'ordinal',
    'CARDINAL': 'cardinal',
    # 'PERCENT': 'number',
    # 'MONEY': 'number',
    # 'QUANTITY': 'number',
    # 'ORDINAL': 'number',
    # 'CARDINAL': 'number',
    'LAW': 'law'
}

  def __init__(self, DATASET, DOMAIN, PAIRS):
    self.MAX_NB_WORDS = 20000
    self.VALIDATION_SPLIT = 0.8
    self.DIR = 'data/processed' # where will be exported
    self.DATASET=DATASET
    self.DOMAIN=DOMAIN
    self.PAIRS = PAIRS
    self.nlp = spacy.load('en_core_web_sm')

  def read_pairs(self, df):
    bug_pairs = []
    bug_ids = set()
    for row in df.iterrows():
      duplicates = row[1]['duplicate']
      bug1 = row[1]['issue_id']
      duplicates = [] if (type(duplicates) == float) else np.array(duplicates.split(';'), dtype=np.float)
      if len(duplicates) == 0: # No duplicate
        bug_ids.add(int(bug1))
      else: # duplicate
        bug_ids.add(int(bug1))
        for bug2 in duplicates:
          bug_pairs.append((int(bug1), int(bug2)))
          bug_ids.add(int(bug2))
    with open(os.path.join(self.DIR, 'bug_pairs.txt'), 'w') as f:
      for pair in bug_pairs:
        f.write("%d %d\n" % pair)
    bug_ids = sorted(bug_ids)
    with open(os.path.join(self.DIR, 'bug_ids.txt'), 'w') as f:
      for bug_id in bug_ids:
        f.write("%d\n" % bug_id)
    return bug_pairs, bug_ids

  def split_train_test(self, bug_pairs, VALIDATION_SPLIT):
    random.shuffle(bug_pairs)
    split_idx = int(len(bug_pairs) * VALIDATION_SPLIT)
    with open(os.path.join(self.DIR, 'train.txt'), 'w') as f:
      for pair in bug_pairs[:split_idx]:
        f.write("%d %d\n" % pair)
    test_data = {}
    for pair in bug_pairs[split_idx:]:
      bug1 = int(pair[0])
      bug2 = int(pair[1])
      if bug1 not in test_data:
        test_data[bug1] = set()
      test_data[bug1].add(bug2)
    with open(os.path.join(self.DIR, 'test.txt'), 'w') as f:
      for bug in test_data.keys():
        f.write("{} {}\n".format(bug, ' '.join([str(x) for x in test_data[bug]])))
    print('Train and test created')

  def func_name_tokenize(self, text):
    s = []
    for i, c in enumerate(text):
      if c.isupper() and i > 0 and text[i-1].islower():
        s.append(' ')
      s.append(c)
    return ''.join(s).strip()

  def ner(self, text):
    corpus = self.nlp(text)
    for row in corpus.ents:
      text = text.replace(row.text, self.ENTITY_ENUM[row.label_])
    return text

  def normalize_text(self, text):
    #try:
    text = re.sub(r'\d+((\s\d+)+)?', '', str(text))
    text = self.ner(text)
    tokens = re.compile(r'[\W_]+', re.UNICODE).split(text)
    text = ' '.join([self.func_name_tokenize(token) for token in tokens])
    #except:
    #  return 'description'
    return ' '.join([word.lower() for word in nltk.word_tokenize(text)])

  def save_dict(set, filename):
    with open(os.path.join(args.data, filename), 'w') as f:
      for i, item in enumerate(set):
        f.write('%s\t%d\n' % (item, i))

  def load_dict(filename):
    dict = {}
    with open(os.path.join(args.data, filename), 'r') as f:
      for line in f:
        tokens = line.split('\t')
        dict[tokens[0]] = tokens[1]
    return dict

  def normalized_data(self, bug_ids, df):
    print("Normalizing text...")
    products = set()
    bug_severities = set()
    priorities = set()
    versions = set()
    components = set()
    bug_statuses = set()
    text = []
    normalized_bugs = open(os.path.join(self.DIR, 'normalized_bugs.json'), 'w')
    with tqdm(total=df.shape[0]) as loop:
      for row in df.iterrows():
          bug = row[1]
          products.add(bug['product'])
          bug_severities.add(bug['bug_severity'])
          priorities.add(bug['priority'])
          versions.add(bug['version'])
          components.add(bug['component'])
          bug_statuses.add(bug['bug_status'])
          bug['description'] = self.normalize_text(bug['description'])
          if 'title' in bug:
              bug['title'] = self.normalize_text(bug['title'])
          else:
              bug['title'] = ''
          
          normalized_bugs.write('{}\n'.format(bug.to_json()))

          text.append(bug['description'])
          text.append(bug['title'])
          loop.update(1)
    save_dict(products, os.path.join(self.DIR, 'product.dic'))
    save_dict(bug_severities, os.path.join(self.DIR, 'bug_severity.dic'))
    save_dict(priorities, os.path.join(self.DIR, 'priority.dic'))
    save_dict(versions, os.path.join(self.DIR, 'version.dic'))
    save_dict(components, os.path.join(self.DIR, 'component.dic'))
    save_dict(bug_statuses, os.path.join(self.DIR, 'bug_status.dic'))
    return text

  def build_vocabulary(self, train_text, MAX_NB_WORDS):
    word_freq = self.build_freq_dict(train_text)
    print('word vocabulary')
    word_vocab = self.save_vocab(word_freq, MAX_NB_WORDS, 'word_vocab.pkl')
    return word_vocab

  def build_freq_dict(self, train_text):
    print('building frequency dictionaries')
    word_freq = defaultdict(int)
    for text in tqdm(train_text):
      for word in text.split():
        word_freq[word] += 1
    return word_freq

  def save_vocab(self, freq_dict, vocab_size, filename):
    top_tokens = sorted(freq_dict.items(), key=lambda x: -x[1])[:vocab_size - 2]
    print('most common token is %s which appears %d times' % (top_tokens[0][0], top_tokens[0][1]))
    print('less common token is %s which appears %d times' % (top_tokens[-1][0], top_tokens[-1][1]))
    vocab = {}
    i = 2  # 0-index is for padding, 1-index is for UNKNOWN
    for j in range(len(top_tokens)):
      vocab[top_tokens[j][0]] = i
      i += 1
    with open(os.path.join(self.DIR, filename), 'wb') as f:
      pickle.dump(vocab, f)
    return vocab

  def load_vocab(self, filename):
      with open(os.path.join(self.DIR, filename), 'rb') as f:
          return pickle.load(f)

  def dump_bugs(self, word_vocab, total):
      bug_dir = os.path.join(self.DIR, 'bugs')
      if not os.path.exists(bug_dir):
          os.mkdir(bug_dir)
      bugs = []
      print("Reading the normalized_bugs.json ...")
      product_dict = load_dict(os.path.join(self.DIR,'product.dic'))
      bug_severity_dict = load_dict(os.path.join(self.DIR,'bug_severity.dic'))
      priority_dict = load_dict(os.path.join(self.DIR,'priority.dic'))
      version_dict = load_dict(os.path.join(self.DIR,'version.dic'))
      component_dict = load_dict(os.path.join(self.DIR,'component.dic'))
      bug_status_dict = load_dict(os.path.join(self.DIR,'bug_status.dic'))
      with open(os.path.join(self.DIR, 'normalized_bugs.json'), 'r') as f:
          #loop = tqdm(f)
          with tqdm(total=total) as loop:
              for line in f:
                  bug = json.loads(line)
                  bug['product'] = product_dict[bug['product']]
                  bug['bug_severity'] = bug_severity_dict[bug['bug_severity']]
                  bug['priority'] = priority_dict[bug['priority']]
                  bug['version'] = version_dict[bug['version']]
                  bug['component'] = component_dict[bug['component']]
                  bug['bug_status'] = bug_status_dict[bug['bug_status']]
                  bugs.append(bug)
                  loop.update(1)

      return bugs

  def dump_vocabulary(self, bugs, word_vocab, bug_dir):
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

  def processing_dump(self, bugs, word_vocab):
      #clear_output()
      cpu = os.cpu_count() - 1
      pool = Pool(processes=cpu) # start 4 worker processes
      bug_dir = os.path.join(self.DIR, 'bugs')
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
              works.append(pool.apply_async(self.dump_vocabulary, (s, word_vocab, bug_dir, )))
              #dump_vocabulary(s, bug_dir)

      print("Executing the works...")
      res = [w.get() for w in works]

      # dump_vocabulary(bugs, word_vocab, bug_dir)

      print("All done!")

  def run(self):
    
      # create dataset directory
      bug_dir = os.path.join(self.DIR, self.DATASET)
      if not os.path.exists(bug_dir):
          os.mkdir(bug_dir)

      normalized = os.path.join('data/normalized', self.DATASET)

      self.DIR = bug_dir
      self.DOMAIN = os.path.join(normalized, self.DOMAIN)
      self.PAIRS = os.path.join(normalized, self.PAIRS)
      
      # Train
      df_train = pd.read_csv('{}.csv'.format(self.DOMAIN))
      df_train.columns = ['issue_id','bug_severity','bug_status','component',
                          'creation_ts','delta_ts','description','dup_id','priority',
                          'product','resolution','title','version']

      ### Pairs
      df_train_pair = pd.read_csv('{}.csv'.format(self.PAIRS))

      bug_pairs, bug_ids = self.read_pairs(df_train_pair)
      print("Number of bugs: {}".format(len(bug_ids)))
      print("Number of pairs: {}".format(len(bug_pairs)))

      # Split into train/test
      self.split_train_test(bug_pairs, self.VALIDATION_SPLIT)
      # Normalize the text
      text = self.normalized_data(bug_ids, df_train)
      # Build the vocab
      word_vocab = self.build_vocabulary(text, self.MAX_NB_WORDS)
      
      # Dump the preprocessed bugs
      num_lines =  len(open(os.path.join(self.DIR, 'normalized_bugs.json'), 'r').read().splitlines()) * 2
      total = num_lines // 2
      bugs = self.dump_bugs(word_vocab, total)
      self.processing_dump(bugs, word_vocab)
      print("Saved!")

def main():

    try:
      _, dataset = sys.argv
    except:
      print('###### Missing the dataset to be processed #########')
      print("Ex: $ python script.py {eclipse, eclipse_small, netbeans, openoffice} ")
      exit(1)
    
    op = {
      'eclipse' : {
        'DATASET' : 'eclipse',
        'DOMAIN' : 'eclipse',
        'PAIRS' : 'eclipse_pairs'
      },
      'eclipse_small' : {
        'DATASET' : 'eclipse',
        'DOMAIN' : 'eclipse_small',
        'PAIRS' : 'eclipse_small_pairs'
      },
      'netbeans' : {
        'DATASET' : 'netbeans',
        'DOMAIN' : 'netbeans',
        'PAIRS' : 'netbeans_pairs'
      },
      'openoffice' : {
        'DATASET' : 'openoffice',
        'DOMAIN' : 'openOffice',
        'PAIRS' : 'openOffice_pairs'
      }
    }

    preprocessing = Preprocess(op[dataset]['DATASET'], op[dataset]['DOMAIN'], op[dataset]['PAIRS'])
    preprocessing.run()

if __name__ == '__main__':
  main()