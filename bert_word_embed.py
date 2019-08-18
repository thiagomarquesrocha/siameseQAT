#!/usr/bin/env python
# coding: utf-8

# # Bert pretrained in the dataset
# 
# https://pypi.org/project/bert-embedding/

# ## Word embedding vocabulary

# In[1]:

from multiprocessing import Pool
import fasttext
import re
import numpy as np
import pandas as pd

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

# In[2]:
from methods.baseline import Baseline
import _pickle as pickle

import mxnet as mx
from bert_embedding import BertEmbedding
# In[3]:


MAX_SEQUENCE_LENGTH_T = 100 # 40
MAX_SEQUENCE_LENGTH_D = 200 # 200
EMBEDDING_DIM = 300
MAX_NB_WORDS = 2000


# In[4]:


# Domain to use
DOMAIN = 'eclipse'
# Dataset paths
DIR = 'data/processed/{}'.format(DOMAIN)
DIR_PAIRS = 'data/normalized/{}'.format(DOMAIN)
DATASET = os.path.join('data/normalized/{}'.format(DOMAIN), '{}.csv'.format(DOMAIN))

def load_bugs(baseline):   
    removed = []
    baseline.corpus = []
    baseline.sentence_dict = {}
    baseline.bug_set = {}
    title_padding, desc_padding = [], []
    for bug_id in tqdm(baseline.bug_ids):
        try:
            bug = pickle.load(open(os.path.join(baseline.DIR, 'bugs', '{}.pkl'.format(bug_id)), 'rb'))
            baseline.bug_set[bug_id] = bug
            #break
        except:
            removed.append(bug_id)
    
    if len(removed) > 0:
        for x in removed:
            baseline.bug_ids.remove(x)
        baseline.removed = removed
        print("{} were removed. To see the list call self.removed".format(len(removed)))

def run():
    # In[5]:


    baseline = Baseline(DIR, DATASET, MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D)


    # In[6]:


    baseline.load_ids(DIR)
    print(len(baseline.bug_ids))


    # In[8]:


    # #### Read the corpus from bugs
    load_bugs(baseline)
    # In[9]:


    sent_title = [baseline.bug_set[bug_id]['title'][:MAX_SEQUENCE_LENGTH_T] for bug_id in baseline.bug_ids]
    sent_desc = [baseline.bug_set[bug_id]['description'][:MAX_SEQUENCE_LENGTH_D] for bug_id in baseline.bug_ids]


    # In[10]:


    print(len(sent_title), len(sent_desc))


    # ### BERT embedding

    # In[11]:

    ctx = mx.gpu(0)
    bert_embedding = BertEmbedding(ctx, batch_size=32, max_seq_length=MAX_SEQUENCE_LENGTH_D)


    # ### Save dataset vocabulary embedding
    # In[23]:

    # res = paralelize_processing([baseline.bug_ids, sent_title, sent_desc], 
    #                                     vectorizing_bugs, (baseline.DIR, bert_embedding, baseline.bug_set, ))

    vectorizing_bugs([baseline.bug_ids, sent_title, sent_desc], baseline.DIR, bert_embedding, baseline.bug_set)
    print("Vectorized all dataset with BERT.")
    # In[ ]:


    bug_selected = np.random.choice(baseline.bug_ids, 1)[0]

    bug = baseline.bug_set[bug_selected]
    
    print("Testing if a random bug has bert embeddings")

    assert len(bug['title_bert_embed']) == 768
    assert len(bug['desc_bert_embed']) == 768

    print("Embedding with BERT trained!")


def vectorizing_bugs(batch, DIR, bert_embedding, bug_set):
    #diff = 3208 + 1522
    bug_ids, sent_title, sent_desc = batch
    print("Starting vectorizer with BERT")
    loop = tqdm(total=len(bug_ids))
    loop.set_description('Vectorizing with BERT')
    index = 0
    for title, desc, bug_id in zip(sent_title, sent_desc, bug_ids):
        #if index < diff: pass
        result_title = bert_embedding(title, 'avg')
        result_desc = bert_embedding(desc, 'avg')

        bug = bug_set[bug_id]
        bug['title_bert_embed'] = np.mean(result_title[0][1], 0)
        bug['desc_bert_embed'] = np.mean(result_desc[0][1], 0)
        
        with open(os.path.join(DIR, 'bugs', '{}.pkl'.format(bug_id)), 'wb') as f:
            pickle.dump(bug, f)
        loop.update(1)
        index+=1
    loop.close()
    print("Done vectorization with BERT")

def paralelize_processing(array, callback, parameters):
      cpu = os.cpu_count() // 2
      pool = Pool(processes=cpu) # start N worker processes
      works = []
      n = len(array[0]) // cpu
      n = 1 if n == 0 else n
      sliced = []
      pos_end = n
      end = len(array[0])
      for i in range(cpu):
          pos_end = end if pos_end>=end else pos_end
          pos_end = end if (i+1) == cpu and pos_end < end else pos_end
          sliced.append((array[0][i*n:pos_end], array[1][i*n:pos_end], array[2][i*n:pos_end]))
          pos_end += n

      print("Slicing in {} workers".format(len(sliced)))
      for s in sliced:
          if len(s) > 0:
              config = list(parameters)
              config.insert(0, s[0]) # bug ids
              config.insert(1, s[1]) # titles
              config.insert(2, s[2]) # descs
              config = tuple(config)
              works.append(pool.apply_async(callback, config))
              #dump_vocabulary(s, bug_dir)

      print("Executing the works...")
      res = [w.get() for w in works]
      return res

if __name__ == '__main__':
    run()

