import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from baseline import Baseline
from keras.layers import Conv1D, Input, Add, Activation, Dropout, Embedding, \
        MaxPooling1D, GlobalMaxPool1D, Flatten, Dense, Concatenate, BatchNormalization
from keras.models import Model

class Retrieval():
    def __init__(self):
        pass
    
    def load_bugs(self, data, train):
        self.baseline.load_ids(data)
        bug_dir = os.path.join(data)
        self.baseline.prepare_dataset(bug_dir)
        self.baseline.load_bugs()

    def create_bucket(self, df):
        print("Creating the buckets...")
        buckets = {}
        loop = tqdm(total=df.shape[0])
        for row in df.iterrows():
            name = row[1]['issue_id']
            duplicates = row[1]['duplicate']
            duplicates = [] if (type(duplicates) == float) else np.array(str(duplicates).split(';'), int)
            buckets[name] = duplicates
            loop.update(1)
        loop.close()
        self.buckets = buckets

    def create_queries(self, path_test):
        print("Creating the queries...")
        test = []
        with open(path_test, 'r') as file_test:
            for row in tqdm(file_test):
                duplicates = np.array(row.split(' '), int)
                # Create the test queries
                query = duplicates[0]
                duplicates = np.delete(duplicates, 0)
                while duplicates.shape[0] > 0:
                    dup = duplicates[0]
                    duplicates = np.delete(duplicates, 0)
                    test.append([query, dup])
        self.test = test

    def read_model(self, MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D):
        
        name = 'baseline_1000epoch_10steps_512batch(eclipse)'
        similarity_model = Baseline.load_model('', name, {'l2_normalize' : Baseline.l2_normalize})

        bug_t = Input(shape = (MAX_SEQUENCE_LENGTH_T, ), name = 'title')
        bug_d = Input(shape = (MAX_SEQUENCE_LENGTH_D, ), name = 'desc')
        # Encoder
        title_encoder = similarity_model.get_layer('FeatureLstmGenerationModel')
        desc_encoder = similarity_model.get_layer('FeatureCNNGenerationModel')
        # Bug feature
        bug_encoded_t = title_encoder(bug_t)
        bug_encoded_d = desc_encoder(bug_d)

        model = similarity_model.get_layer('merge_features_in')
        output = model([bug_encoded_t, bug_encoded_d])

        model = Model(inputs=[bug_t, bug_d], outputs=[output])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        
        self.model = model

    def infer_vector(self, bugs, path_data):
        bug_set = self.baseline.get_bug_set()
        with open(path_data, 'r') as file_train:
            for row in file_train:
                dup_a_id, dup_b_id = np.array(row.split(' '), int)
                if dup_a_id not in bug_set or dup_b_id not in bug_set: continue
                bug_a = bug_set[dup_a_id]
                bug_b = bug_set[dup_b_id]
                bug_a_vector = self.model.predict([bug_a['title_word'], bug_a['description_word']])
                bug_b_vector = self.model.predict([bug_b['title_word'], bug_b['description_word']])
                bugs.append(bug_a_vector)
                bugs.append(bug_b_vector)

    def run(self, path, path_buckets, path_train, path_test):

        MAX_SEQUENCE_LENGTH_T = 100 # Title
        MAX_SEQUENCE_LENGTH_D = 100 # Description
        #DIR = 'data/processed/eclipse'

        # Create the instance from baseline
        self.baseline = Baseline(path, MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D)

        df = pd.read_csv(path_buckets)

        self.train_bugs = []

        # Load bug ids
        self.load_bugs(path, path_train)
        # Create the buckets
        self.create_bucket(df)
        # Read and create the test queries duplicate
        self.create_queries(path_test)
        # Read the siamese model
        self.read_model(MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D)
        # Infer vector to all train
        self.infer_vector(self.train_bugs, path_train)
        # Infer vector to all test
        # Indexing all train in KNN method
        # Recommend neighborhood instances from test sample
        # Generating the rank result

if __name__ == '__main__':
    retrieval = Retrieval()
    retrieval.run(
        'data/processed/eclipse', 
        'data/normalized/eclipse/eclipse_pairs.csv', 
        'data/processed/eclipse/train.txt', 
        'data/processed/eclipse/test.txt')
    print("Retrieval")
