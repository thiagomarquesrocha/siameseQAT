import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

# from methods.baseline import Baseline
# from keras.layers import Conv1D, Input, Add, Activation, Dropout, Embedding, \
#         MaxPooling1D, GlobalMaxPool1D, Flatten, Dense, Concatenate, BatchNormalization
# from keras.models import Model
from sklearn.neighbors import NearestNeighbors
from operator import itemgetter

class Retrieval():
    def __init__(self):
        pass
    
    def load_bugs(self, data, train):
        self.baseline.load_ids(data)
        self.baseline.prepare_dataset()
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
            buckets[name].add(name)
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
                '''
                Ex: Netbeans bug 97781 point to 67568 (does not exist)
                '''
                if query.shape[0] <= 0: # when the duplicate does not exist
                    buckets[bucket_name] = set() # set the bucket alone
                    buckets[bucket_name].add(bucket_name)
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
        self.buckets = buckets

    def create_queries(self):
        self.test = self.baseline.test_data

    def get_buckets_for_bugs(self):
        issues_by_buckets = {}
        for bucket in tqdm(self.buckets):
            issues_by_buckets[bucket] = bucket
            for issue in np.array(self.buckets[bucket]).tolist():
                issues_by_buckets[issue] = bucket
        return issues_by_buckets

    def infer_vector_train(self, bugs):
        bug_set = self.baseline.get_bug_set()
        bug_unique = set()
        for row in tqdm(bugs):
            dup_a_id, dup_b_id = row
            bug_unique.add(dup_a_id)
            bug_unique.add(dup_b_id)
        self.bugs_train = bug_unique

    def infer_vector_test(self, bugs, result):
        bug_set = self.baseline.get_bug_set()
        print("Selecting buckets duplicates...")
        buckets_duplicates = [key for key in tqdm(self.buckets) if len(self.buckets[key]) > 1]
        test_no_present_in_trained = []
        print("Selecting only bugs did not used in the train...")
        for row in tqdm(bugs):
            dup_a_id, dup_b_id = row
            diff = list(set(row) - self.bugs_train)
            test_no_present_in_trained += diff
        print("Formating the rank result the retrieval model...")
        queries = []
        for bug_id in test_no_present_in_trained:
            queries += [(bug_id, master_id) for master_id in buckets_duplicates]
        last_query = { 'bug_id' : -1, 'master_id' : -1 }
        rank = []
        for bug_id, master_id in tqdm(queries):
            bug = bug_set[bug_id]
            master = bug_set[master_id]
            if bug_id != last_query['bug_id']:
                if len(rank) > 0:
                    rank=sorted(rank, key = itemgetter(1), reverse = True)
                    result.append({ 'rank' : rank, 
                                    'dup_a' : last_query['bug_id'],
                                    'dup_b' : last_query['master_id'] })
                rank = []
            bug_vector = self.model.predict([ [bug['title_word']], [master['title_word']], 
                                        [bug['description_word']], [master['description_word']],
                                        [self.get_info(bug)], [self.get_info(master)] ])[0]
            rank.append((master_id, bug_vector[1]))
            last_query['master_id'] = master_id
            last_query['bug_id'] = bug_id

    def get_info(self, bug):
        if self.baseline.DOMAIN != 'firefox':
            info = np.concatenate((
                self.baseline.to_one_hot(bug['bug_severity'], self.baseline.info_dict['bug_severity']),
                self.baseline.to_one_hot(bug['bug_status'], self.baseline.info_dict['bug_status']),
                self.baseline.to_one_hot(bug['component'], self.baseline.info_dict['component']),
                self.baseline.to_one_hot(bug['priority'], self.baseline.info_dict['priority']),
                self.baseline.to_one_hot(bug['product'], self.baseline.info_dict['product']),
                self.baseline.to_one_hot(bug['version'], self.baseline.info_dict['version']))
            )
        else:
            info = np.concatenate((
                self.baseline.to_one_hot(bug['bug_status'], self.baseline.info_dict['bug_status']),
                self.baseline.to_one_hot(bug['component'], self.baseline.info_dict['component']),
                self.baseline.to_one_hot(bug['priority'], self.baseline.info_dict['priority']),
                self.baseline.to_one_hot(bug['version'], self.baseline.info_dict['version']))
            )
        return info

    def run(self, path, dataset, path_buckets, path_train, path_test):
        pass
        # MAX_SEQUENCE_LENGTH_T = 100 # Title
        # MAX_SEQUENCE_LENGTH_D = 100 # Description

        # # Create the instance from baseline
        # self.baseline = Baseline(path, dataset, MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D)

        # df = pd.read_csv(path_buckets)

        # # Load bug ids
        # self.load_bugs(path, path_train)
        # # Create the buckets
        # self.create_bucket(df)
        # # Read and create the test queries duplicate
        # self.create_queries(path_test)
        # # Read the siamese model
        # self.read_model(MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D)
        
        # self.train_vectorized, self.test_vectorized = [], []
        # self.bug_set_cluster_train, self.bug_set_cluster_test = [], []
        # self.read_train(path_train)
        # # Infer vector to all train
        # self.create_bug_clusters(self.bug_set_cluster_train, self.train)
        # self.infer_vector(self.train, self.train_vectorized)
        # # Infer vector to all test
        # self.create_bug_clusters(self.bug_set_cluster_test, self.test)
        # self.infer_vector(self.test, self.test_vectorized)
        # # Indexing all train in KNN method
        # X = np.array(self.train_vectorized)
        # nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X)
        # # Next we find k nearest neighbor for each point in object X.
        # distances, indices = nbrs.kneighbors(X)
        # # Recommend neighborhood instances from test sample
        # X_test = self.test_vectorized
        # distances_test, indices_test = nbrs.kneighbors(X_test)
        # # Generating the rank result

if __name__ == '__main__':
    retrieval = Retrieval()
    retrieval.run(
        'data/processed/eclipse', 
        'eclipse.csv',
        'data/normalized/eclipse/eclipse.csv', 
        'data/processed/eclipse/train.txt', 
        'data/processed/eclipse/test.txt')
    print("Retrieval")
