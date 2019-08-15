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
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
# Solution from https://github.com/explosion/spaCy/issues/3608
class EntityMatcher(object):
    #name = "entity_matcher"

    def __init__(self, name, nlp, terms, label):
        self.name = name
        patterns = [nlp.make_doc(text) for text in terms]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add(label, None, *patterns)

    def __call__(self, doc):
        matches = self.matcher(doc)
        seen_tokens = set()
        new_entities = []
        entities = doc.ents
        for match_id, start, end in matches:
        #    span = Span(doc, start, end, label=match_id)
        #    doc.ents = list(doc.ents) + [span]
            # check for end - 1 here because boundaries are inclusive
            if start not in seen_tokens and end - 1 not in seen_tokens:
                new_entities.append(Span(doc, start, end, label=match_id))
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))

        doc.ents = tuple(entities) + tuple(new_entities)
        return doc

class Preprocess:

  ENTITY_ENUM = {
    '': 'unknown',
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
    self.nlp = spacy.load('en_core_web_lg')
    self.bugs = {}
    self.bugs_saved = []
    self.TRAIN_PATH = 'train_chronological'
    self.TEST_PATH = 'test_chronological'

    self.improve_ner(self.nlp)

  def read_pairs(self, df):
    bug_pairs = []
    bucket_dups = []
    bug_ids = set()
    buckets = self.create_bucket(df)
    # buckets
    for key in buckets:
      if len(buckets[key]) > 1:
          bucket_dups.append([key, list(buckets[key])])

    bug_pairs, bug_ids = self.getting_pairs(bucket_dups)

    with open(os.path.join(self.DIR, 'bug_pairs.txt'), 'w') as f:
      for pair in bug_pairs:
        f.write("{} {}\n".format(pair[0], pair[1]))
    bug_ids = sorted(bug_ids)
    with open(os.path.join(self.DIR, 'bug_ids.txt'), 'w') as f:
      for bug_id in bug_ids:
        f.write("%d\n" % bug_id)
    return bug_pairs, bug_ids

  def split_train_test(self, bug_pairs, VALIDATION_SPLIT):
    #random.shuffle(bug_pairs)
    split_idx = int(len(bug_pairs) * VALIDATION_SPLIT)
    with open(os.path.join(self.DIR, '{}.txt'.format(self.TRAIN_PATH)), 'w') as f:
      for pair in bug_pairs[:split_idx]:
        f.write("{} {}\n".format(pair[0], pair[1]))
    test_data = {}
    for pair in bug_pairs[split_idx:]:
      bug1 = int(pair[0])
      bug2 = int(pair[1])
      if bug1 not in test_data:
        test_data[bug1] = set()
      test_data[bug1].add(bug2)
    with open(os.path.join(self.DIR, '{}.txt'.format(self.TEST_PATH)), 'w') as f:
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

  def improve_ner(self, nlp):
    # Dates
    dates = []
    for year in range(2000, 2012):
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Oct', 'Nov', 'Dec']:
            for day in range(32):
                dates.append( u'{} {}, {}'.format(day, month, year))
    
    list_terms = [dates,
              (u'Oracle Corporation', u'Oracle', u'Mozilla', u'Google'),
              (u"the Java", u"Java", u"java", u"the java", u"Javadoc", u'API', 
               u"The Javadoc", u"the Javadoc", u"C++", u'c++', u'C/C++', u'XML', u'xml', u'SQL', u'sql',
               u'HTML5', u'HTTP', u'html', u'http', u'html5' u'html 5', u'HTML 5'), 
              (u"Dennis", u"Bob", u"Kamil", u'Kamil Ignacak'), 
              (u'WAR', u'zip'),
              (u'MacOS', u'MacOS X', u'MacOS x', u'Mac OS X', u'Redhat Linux', u'RedHat Enterprise', u'Linux', 
               u'Eclipse', u'eclipse', u'The Eclipse', u'WindowsXP', u'Windows XP', u'Java Virtual Machine', 
               u'VM', u'BIRT', u'Birt Web project', u'Birt', u'Birt Charting', u'JIRA', u'linux',
               u'CDT', u'JREs', u'JRE', u'jre', u'Windows NT', u'SWT', u'CVS', u'Fedora Core',
              u'Tomcat', u'Axis', u'Red Hat', u'GTK'),
              (u'JDK', u'JDT', u'AJNature', u'JavaBuilder', u'AJBuilder', u'OclInvalid', u'Aerogear', 
               u'JSP', u'JGit', u'SDK', u'JEE', u'EPP', u'JEE EPP', u'Widget'),
              (u'1.', u'1)', u'2.', u'2)', u'3.', u'3)',
               u'4.', u'4)', u'5.', u'5)', u'6.', u'6)', u'7.', u'7)',
                u'8.', u'8)', u'9.', u'9)', u'10.', u'10)'),
              (u'ctrl', u'CTRL', u'F1', u'f1', u'F2', u'f2', u'F3', u'f3',
               u'f4', u'F4', u'f5', u'F5', u'f6', u'F6', u'f7', u'F7', u'f8', u'F8',
               u'f9', u'F9', u'f10', u'F10', u'f11', u'F11', u'f12', u'F12', 
               u'CTRL+F1', u'CTRL+F2', u'CTRL+F3', u'CTRL+F4', u'CTRL+F5', u'CTRL+F6',
              u'CTRL+F7', u'CTRL+F8', u'CTRL+F9', u'CTRL+F10', u'CTRL+F11', u'CTRL+F12',
               u'CTRL+TAB', u'ctrl+tab', u'ESC', u'Esc', u'esc', u'CTRL+1', u'CTRL+2', u'CTRL+3', u'CTRL+4',
              u'CTRL+5', u'CTRL+6', u'CTRL+7', u'CTRL+8', u'CTRL+9', u'CTRL+0', u'ctrl+1', u'ctrl+2',
              u'ctrl+3', u'ctrl+4', u'ctrl+5', u'ctrl+6', u'ctrl+7', u'ctrl+8', u'ctrl+9', u'ctrl+0',
              u'crtl + space', u'CTRL + SPACE', u'CTRL + Space', u'CTRL-C', u'CTRL-V', u'ctrl-c', u'ctrl-v')
             ]
    list_labels = ['DATE', 'ORG', "LANGUAGE", "PERSON", "FILE", "PRODUCT", "COMPONENT", "STEP NUMBER", "KEYBOARD"]

    for terms, label in zip(list_terms, list_labels):
        entity_matcher = EntityMatcher(label, nlp, terms, label)
        nlp.add_pipe(entity_matcher, after='ner')

  def ner(self, text):
    corpus = self.nlp(text)
    for row in corpus.ents:
      text = text.replace(row.text, self.ENTITY_ENUM[row.label_] if row.label_ in self.ENTITY_ENUM else row.label_)
    return text

  def normalize_text(self, text):
    #try:
    text = re.sub(r'(bug|Bug) (#|)[0-9]{1,}', 'bug id', str(text)) # bug id
    text = re.sub(r'\w{2,}(.java)', 'java class', text) # .java class files
    text = self.ner(text)
    tokens = re.compile(r'[\W_]+', re.UNICODE).split(text)
    text = ' '.join([self.func_name_tokenize(token) for token in tokens])
    text = re.sub(r'\d+((\s\d+)+)?', ' ', text)
    text = text[:100000] # limit of spacy lib
    #except:
    #  return 'description'
    text = [word.lower() for word in nltk.word_tokenize(text)]
    text = ' '.join([word for word in text]).encode('utf-8')
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
    normalized_bugs_json = []
    print("Total:", df.shape[0])
    res = self.paralelize_processing(df, self.processing_normalized_data, (self.normalize_text, ))
    for result in res:
      products = products.union(result[0])
      bug_severities = bug_severities.union(result[1])
      priorities = priorities.union(result[2])
      versions = versions.union(result[3])
      components = components.union(result[4])
      bug_statuses = bug_statuses.union(result[5])
      text += result[6]
      normalized_bugs_json += result[7]
    print("Total of normalized: ", len(normalized_bugs_json))
    print("Writing the normalized_bugs.json")
    with open(os.path.join(self.DIR, 'normalized_bugs.json'), 'w') as f:
      for row in tqdm(normalized_bugs_json):
        f.write(row)
    
    self.save_dict(products, os.path.join(self.DIR, 'product.dic'))
    self.save_dict(bug_severities, os.path.join(self.DIR, 'bug_severity.dic'))
    self.save_dict(priorities, os.path.join(self.DIR, 'priority.dic'))
    self.save_dict(versions, os.path.join(self.DIR, 'version.dic'))
    self.save_dict(components, os.path.join(self.DIR, 'component.dic'))
    self.save_dict(bug_statuses, os.path.join(self.DIR, 'bug_status.dic'))
    return text

  def processing_normalized_data(self, df, normalize_text):
    products = set()
    bug_severities = set()
    priorities = set()
    versions = set()
    components = set()
    bug_statuses = set()
    text = []
    normalized_bugs_json = []
    with tqdm(total=df.shape[0]) as loop:
      for row in df.iterrows():
          bug = row[1]
          products.add(bug['product'])
          bug_severities.add(bug['bug_severity'])
          priorities.add(bug['priority'])
          versions.add(bug['version'])
          components.add(bug['component'])
          bug_statuses.add(bug['bug_status'])
          
          if 'description' in bug:
              bug['description'] = normalize_text(bug['description'])
         
          if 'title' in bug:
               bug['title'] = normalize_text(bug['title'])
          
          if bug['description'] == '':
              bug['description'] = bug['title']

          if bug['title'] == '':
              bug['title'] = bug['description']

          normalized_bugs_json.append('{}\n'.format(bug.to_json()))

          text.append(bug['description'])
          text.append(bug['title'])
          loop.update(1)
    return [products, bug_severities, priorities, versions, components, bug_statuses, text, normalized_bugs_json]

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
      bugs_set = {}
      bugs_saved = []
      for bug in tqdm(bugs):
          #bug = json.loads(line)
          #print(bug)
          cont+=1
          bug['description_word'] = [word_vocab.get(w.encode('utf-8'), UNK) for w in bug['description'].split()]
          if len(bug['title']) == 0:
              bug['title'] = bug['description'][:10]
          bug['title_word'] = [word_vocab.get(w.encode('utf-8'), UNK) for w in bug['title'].split()]
          #bug.pop('description')
          #bug.pop('title')
          bugs_set[bug['issue_id']] = bug
          with open(os.path.join(bug_dir, str(bug['issue_id']) + '.pkl'), 'wb') as f:
              pickle.dump(bug, f)
          bugs_saved.append(bug['issue_id'])

      return [bugs_set, bugs_saved]

  def paralelize_processing(self, bugs, callback, parameters):
      cpu = os.cpu_count() - 1
      pool = Pool(processes=cpu) # start N worker processes
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

      print("Slicing in {} workers".format(len(sliced)))
      for s in sliced:
          if len(s) > 0:
              config = list(parameters)
              config.insert(0, s)
              config = tuple(config)
              works.append(pool.apply_async(callback, config))
              #dump_vocabulary(s, bug_dir)

      print("Executing the works...")
      res = [w.get() for w in works]
      return res

  def processing_dump(self, bugs, word_vocab, bugs_id, bugs_id_dataset):
      #clear_output()
      bug_dir = os.path.join(self.DIR, 'bugs')
      res = self.paralelize_processing(bugs, self.dump_vocabulary, (word_vocab, bug_dir, ))
      for result in res:
        bugs_set = result[0]
        bugs_saved = result[1]
        for bug in bugs_set:
          self.bugs[bug] = bugs_set[bug]
        self.bugs_saved += bugs_saved
      #self.dump_vocabulary(bugs, word_vocab, bug_dir)

      self.validing_bugs_id(bugs_id, bugs_id_dataset)

      print("All done!")

  def validing_bugs_id(self, bugs_id, bugs_id_dataset):
      print("Check if all bugs id regirested in the pairs exist in dataset")
      bugs_invalid = set(bugs_id) - set(bugs_id_dataset)
      bugs_id_dataset = set(bugs_id_dataset) - bugs_invalid
      bugs_id_dataset = sorted(bugs_id_dataset)
      with open(os.path.join(self.DIR, 'bug_ids.txt'), 'w') as f:
        for bug_id in bugs_id_dataset:
          f.write("%d\n" % bug_id)
      print("Bugs not present in dataset: ", list(bugs_invalid))
      bug_pairs = []
      with open(os.path.join(self.DIR, '{}.txt'.format(self.TRAIN_PATH)), 'r') as f:
          for line in f:
              bug1, bug2 = line.strip().split()
              if bug1 not in bugs_invalid and bug2 not in bugs_invalid:
                bug_pairs.append([bug1, bug2])
      with open(os.path.join(self.DIR, '{}.txt'.format(self.TRAIN_PATH)), 'w') as f:
          for pairs in bug_pairs:
              f.write("{} {}\n".format(pairs[0], pairs[1]))

  def create_bucket(self, df):
    print("Creating the buckets...")
    buckets = {}
    # Reading the buckets
    df_buckets = df[df['dup_id'] == '[]']
    loop = tqdm(total=df_buckets.shape[0])
    for row in df_buckets.iterrows():
        name = row[1]['issue_id']
        buckets[name] = set()
        buckets[name].add(name)
        loop.update(1)
    loop.close()
    # Fill the buckets
    df_duplicates = df[df['dup_id'] != '[]']
    loop = tqdm(total=df_duplicates.shape[0])
    for row_bug_id, row_dup_id in df_duplicates[['issue_id', 'dup_id']].values:
        bucket_name = int(row_dup_id)
        dup_id = row_bug_id
        while bucket_name not in buckets:
            query = df_duplicates[df_duplicates['issue_id'] == bucket_name]
            if query.shape[0] <= 0: 
                break
            bucket_name = int(query['dup_id'])
        '''
            Some bugs duplicates point to one master that
            does not exist in the dataset like openoffice master=152778
        '''
        if bucket_name in buckets:
            buckets[bucket_name].add(dup_id)
        loop.update(1)
    loop.close()
    return buckets

  def getting_pairs(self, array):
      res = []
      bug_ids = set()
      for row in array:
          dup_bucket, dups = row
          bug_ids.add(dup_bucket)
          dups = list(dups)
          while len(dups) > 1:
              bucket = dups[0]
              bug_ids.add(bucket)
              dups.remove(bucket)
              for d in dups:
                  bug_ids.add(d)
                  res.append([bucket, d])
      return res, bug_ids    
  
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
      #df_train_pair = pd.read_csv('{}.csv'.format(self.PAIRS))

      bug_pairs, bug_ids = self.read_pairs(df_train)
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