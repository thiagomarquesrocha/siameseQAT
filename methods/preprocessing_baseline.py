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
import os.path
from os import path

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
    # 'PERCENT': 'percent',
    # 'MONEY': 'money',
    # 'QUANTITY': 'quantity',
    # 'ORDINAL': 'ordinal',
    # 'CARDINAL': 'cardinal',
    'PERCENT': 'number',
    'MONEY': 'number',
    'QUANTITY': 'number',
    'ORDINAL': 'number',
    'CARDINAL': 'number',
    'LAW': 'law'
}

  def __init__(self, DATASET, DOMAIN, PAIRS, COLAB):
    self.MAX_NB_WORDS = 20000
    self.VALIDATION_SPLIT = 0.9
    self.COLAB = COLAB
    self.DIR = '{}data/processed'.format(COLAB) # where will be exported
    self.DATASET=DATASET
    self.DOMAIN=DOMAIN
    self.PAIRS = PAIRS
    self.nlp = spacy.load('en_core_web_sm')
    self.bugs = {}
    self.bugs_saved = []

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
    tokens = re.compile(r'[\W_]+', re.UNICODE).split(str(text))
    text = ' '.join([self.func_name_tokenize(token) for token in tokens])
    text = re.sub(r'\d+((\s\d+)+)?', ' ', text)
    text = text[:100000] # limit of spacy lib
    text = self.ner(text)
    #except:
    #  return 'description'
    text = [word.lower() for word in nltk.word_tokenize(text)]
    text = ' '.join([word for word in text if len(word) > 1])
    return text

  def save_dict(self, set, filename):
    with open(filename, 'w') as f:
      for i, item in enumerate(set):
        f.write('%s\t%d\n' % (item, i))

  def load_dict(self, filename):
    dict = {}
    with open(filename, 'r') as f:
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
    print("Total:", df.shape[0])
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
    self.save_dict(products, os.path.join(self.DIR, 'product.dic'))
    self.save_dict(bug_severities, os.path.join(self.DIR, 'bug_severity.dic'))
    self.save_dict(priorities, os.path.join(self.DIR, 'priority.dic'))
    self.save_dict(versions, os.path.join(self.DIR, 'version.dic'))
    self.save_dict(components, os.path.join(self.DIR, 'component.dic'))
    self.save_dict(bug_statuses, os.path.join(self.DIR, 'bug_status.dic'))
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
      product_dict = self.load_dict(os.path.join(self.DIR,'product.dic'))
      bug_severity_dict = self.load_dict(os.path.join(self.DIR,'bug_severity.dic'))
      priority_dict = self.load_dict(os.path.join(self.DIR,'priority.dic'))
      version_dict = self.load_dict(os.path.join(self.DIR,'version.dic'))
      component_dict = self.load_dict(os.path.join(self.DIR,'component.dic'))
      bug_status_dict = self.load_dict(os.path.join(self.DIR,'bug_status.dic'))

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
          self.bugs[bug['issue_id']] = bug
          with open(os.path.join(bug_dir, str(bug['issue_id']) + '.pkl'), 'wb') as f:
              pickle.dump(bug, f)
          self.bugs_saved.append(bug['issue_id'])

  def processing_dump(self, bugs, word_vocab, bugs_id, bugs_id_dataset):
      #clear_output()
      #cpu = os.cpu_count() - 1
      #pool = Pool(processes=cpu) # start 4 worker processes
      bug_dir = os.path.join(self.DIR, 'bugs')
      #print("Starting the slice ...")
      # works = []
      # n = len(bugs) // cpu
      # n = 1 if n == 0 else n
      # sliced = []
      # pos_end = n
      # end = len(bugs)
      # for i in range(cpu):
      #     pos_end = end if pos_end>=end else pos_end
      #     pos_end = end if (i+1) == cpu and pos_end < end else pos_end
      #     sliced.append(bugs[i*n:pos_end])
      #     pos_end += n

      # print("Slicing done!")
      # for s in sliced:
      #     if len(s) > 0:
      #         works.append(pool.apply_async(self.dump_vocabulary, (s, word_vocab, bug_dir, )))
      #         #dump_vocabulary(s, bug_dir)

      # print("Executing the works...")
      # res = [w.get() for w in works]

      self.dump_vocabulary(bugs, word_vocab, bug_dir)

      self.validing_bugs_id(bugs_id, bugs_id_dataset)

      print("All done!")

  def validing_bugs_id(self, bugs_id, bugs_id_dataset):
      print("Check if all bugs id regirested in the pairs exist in dataset")
      bugs_id_dataset = sorted(bugs_id_dataset)
      with open(os.path.join(self.DIR, 'bug_ids.txt'), 'w') as f:
        for bug_id in bugs_id_dataset:
          f.write("%d\n" % bug_id)
      bugs_invalid = set(bugs_id) - set(bugs_id_dataset)
      print("Bugs not present in dataset: ", list(bugs_invalid))
      bug_pairs = []
      with open(os.path.join(self.DIR, 'train.txt'), 'r') as f:
          for line in f:
              bug1, bug2 = line.strip().split()
              if bug1 and bugs_invalid and bug2 not in bugs_invalid:
                bug_pairs.append([bug1, bug2])
      with open(os.path.join(self.DIR, 'train.txt'), 'w') as f:
          for pairs in bug_pairs:
              f.write("{} {}\n".format(pairs[0], pairs[1]))
      
  def run(self):
    
      # create dataset directory
      bug_dir = os.path.join(self.DIR, self.DATASET)
      if not os.path.exists(bug_dir):
          os.mkdir(bug_dir)

      normalized = os.path.join('{}data/normalized'.format(self.COLAB), self.DATASET)

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
      bugs_id_dataset = df_train['issue_id'].values
      print("Number of bugs: {}".format(len(bug_ids)))
      print("Number of pairs: {}".format(len(bug_pairs)))

      # Split into train/test
      self.split_train_test(bug_pairs, self.VALIDATION_SPLIT)

      # Debug
      # test  = [14785, 24843, 32367, 33529]
      # df_train = df_train[df_train['issue_id'].isin(test)]

      # Normalize the text
      text = self.normalized_data(bug_ids, df_train)
      # Build the vocab
      word_vocab = self.build_vocabulary(text, self.MAX_NB_WORDS)
      
      # Dump the preprocessed bugs
      num_lines =  len(open(os.path.join(self.DIR, 'normalized_bugs.json'), 'r').read().splitlines()) * 2
      total = num_lines // 2
      bugs = self.dump_bugs(word_vocab, total)
      self.processing_dump(bugs, word_vocab, bug_ids, bugs_id_dataset)
      print("Saved!")

def main():

    try:
      _, dataset, colab = sys.argv
    except:
      print('###### Missing the dataset to be processed #########')
      print("Ex: $ python preprocessing_baseline.py {eclipse, eclipse_small, netbeans, openoffice} colab")
      exit(1)
    
    if colab == 'colab':
      COLAB = 'drive/My Drive/Colab Notebooks/'
    else:
      COLAB = ''

    op = {
      'eclipse' : {
        'DATASET' : 'eclipse',
        'DOMAIN' : 'eclipse',
        'PAIRS' : 'eclipse_pairs'
      },
      'eclipse_small' : {
        'DATASET' : 'eclipse_small',
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
        'DOMAIN' : 'openoffice',
        'PAIRS' : 'openoffice_pairs'
      }
    }

    preprocessing = Preprocess(op[dataset]['DATASET'], op[dataset]['DOMAIN'], op[dataset]['PAIRS'], COLAB)
    preprocessing.run()

if __name__ == '__main__':
  main()