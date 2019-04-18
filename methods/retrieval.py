import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from methods.baseline import Baseline
from keras.layers import Conv1D, Input, Add, Activation, Dropout, Embedding, \
        MaxPooling1D, GlobalMaxPool1D, Flatten, Dense, Concatenate, BatchNormalization
from keras.models import Model
from sklearn.neighbors import NearestNeighbors

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
        # Reading the buckets
        df_buckets = df[df['dup_id'] == '[]']
        loop = tqdm(total=df_buckets.shape[0])
        for row in df_buckets.iterrows():
            name = row[1]['bug_id']
            buckets[name] = set()
            loop.update(1)
        loop.close()
        # Fill the buckets
        df_duplicates = df[df['dup_id'] != '[]']
        loop = tqdm(total=df_duplicates.shape[0])
        for row_bug_id, row_dup_id in df_duplicates[['bug_id', 'dup_id']].values:
            bucket_name = int(row_dup_id)
            dup_id = row_bug_id
            while bucket_name not in buckets:
                query = df_duplicates[df_duplicates['bug_id'] == bucket_name]
                bucket_name = int(query['dup_id'])
            buckets[bucket_name].add(dup_id)
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

    def read_train(self, path_data):
        self.train = []
        with open(path_data, 'r') as file_train:
            for row in file_train:
                dup_a_id, dup_b_id = np.array(row.split(' '), int)
                self.train.append([dup_a_id, dup_b_id])

    def infer_vector(self, bugs, vectorized):
        bug_set = self.baseline.get_bug_set()
        for row in tqdm(bugs):
            dup_a_id, dup_b_id = row
            # if dup_a_id not in bug_set or dup_b_id not in bug_set: continue
            bug_a = bug_set[dup_a_id]
            bug_b = bug_set[dup_b_id]
            bug_a_vector = self.model.predict([[bug_a['title_word']], [bug_a['description_word']]])[0]
            bug_b_vector = self.model.predict([[bug_b['title_word']], [bug_b['description_word']]])[0]
            vectorized.append(bug_a_vector)
            vectorized.append(bug_b_vector)

    def create_bug_clusters(self, bug_set_cluster, bugs):
        index = 0
        for row in tqdm(bugs):
            dup_a_id, dup_b_id = row
            # if dup_a_id not in bug_set or dup_b_id not in bug_set: continue
            bug_set_cluster[indices[index][:1][0]] = dup_a_id
            bug_set_cluster[indices[index+1][:1][0]] = dup_b_id
            index += 2

    def run(self, path, path_buckets, path_train, path_test):

        MAX_SEQUENCE_LENGTH_T = 100 # Title
        MAX_SEQUENCE_LENGTH_D = 100 # Description

        # Create the instance from baseline
        self.baseline = Baseline(path, MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D)

        df = pd.read_csv(path_buckets)

        # Load bug ids
        self.load_bugs(path, path_train)
        # Create the buckets
        self.create_bucket(df)
        # Read and create the test queries duplicate
        self.create_queries(path_test)
        # Read the siamese model
        self.read_model(MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D)
        
        self.train_vectorized, self.test_vectorized = [], []
        self.bug_set_cluster_train, self.bug_set_cluster_test = [], []
        self.read_train(path_train)
        # Infer vector to all train
        self.create_bug_clusters(self.bug_set_cluster_train, self.train)
        self.infer_vector(self.train, self.train_vectorized)
        # Infer vector to all test
        self.create_bug_clusters(self.bug_set_cluster_test, self.test)
        self.infer_vector(self.test, self.test_vectorized)
        # Indexing all train in KNN method
        X = np.array(self.train_vectorized)
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X)
        # Next we find k nearest neighbor for each point in object X.
        distances, indices = nbrs.kneighbors(X)
        # Recommend neighborhood instances from test sample
        X_test = self.test_vectorized
        distances_test, indices_test = nbrs.kneighbors(X_test)
        # Generating the rank result

if __name__ == '__main__':
    retrieval = Retrieval()
    retrieval.run(
        'data/processed/eclipse', 
        'data/normalized/eclipse/eclipse.csv', 
        'data/processed/eclipse/train.txt', 
        'data/processed/eclipse/test.txt')
    print("Retrieval")
