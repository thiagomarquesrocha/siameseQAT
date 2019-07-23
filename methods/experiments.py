from annoy import AnnoyIndex
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import os

class Experiment:

    def __init__(self, baseline=None, evaluation=None):
        self.baseline = baseline
        self.evaluation = evaluation
    
    def load_ids(self):
        self.baseline.load_ids(self.baseline.DIR)
    
    def load_bugs(self):
        self.baseline.load_bugs()

    def prepare_dataset(self):
        self.baseline.prepare_dataset()

    def retrieval(self, retrieval, baseline, number_of_columns_info, DOMAIN):
        
        self.MAX_SEQUENCE_LENGTH_I = number_of_columns_info # Status, Severity, Version, Component, Module

        # Create the instance from baseline
        path_buckets = 'data/normalized/{}/{}.csv'.format(DOMAIN, DOMAIN)
        retrieval.baseline = baseline

        df = pd.read_csv(path_buckets)

        # Load bug ids
        #retrieval.load_bugs(path, path_train)
        # Create the buckets
        retrieval.create_bucket(df)

        self.retrieval = retrieval
    
    def create_queries(self, path_test):
        print("Creating the queries...")
        test = []
        with open(path_test, 'r') as file_test:
            for row in tqdm(file_test):
                tokens = row.strip().split()
                test.append([int(tokens[0]), [int(bug) for bug in tokens[1:]]])
        self.retrieval.test = test

    def get_buckets_for_bugs(self):
        issues_by_buckets = {}
        for bucket in tqdm(self.retrieval.buckets):
            issues_by_buckets[bucket] = bucket
            for issue in np.array(self.retrieval.buckets[bucket]).tolist():
                issues_by_buckets[issue] = bucket
        return issues_by_buckets
    
    ## Vectorizer model
    def get_model_vectorizer(self, path=None, loaded_model=None):
        if(path):
            loaded_model = load_model(os.path.join("modelos", "model_{}.h5".format(path)))
            
            '''
                {'l2_normalize' : l2_normalize, 
                                        'margin_loss' : margin_loss,
                                        'pos_distance' : pos_distance,
                                        'neg_distance' : neg_distance,
                                        'stack_tensors': stack_tensors}
            '''
        
        return loaded_model
    
    #### Getting the list of candidates
    def indexing_query(self, annoy, queries_test_vectorized, verbose=1):
        X_test = queries_test_vectorized
        distance_test, indices_test = [], []
        loop = enumerate(X_test)
        if(verbose):
            loop = tqdm(enumerate(X_test))
            loop.set_description('Getting the list of candidates from queries')
        for index, row in loop:
            vector = row['vector']
            rank, dist = annoy.get_nns_by_vector(vector, 30, include_distances=True)
            indices_test.append(rank)
            distance_test.append(1 - np.array(dist)) # normalize the similarity between 0 and 1
        if(verbose): loop.close()
        return X_test, distance_test, indices_test
    
    # Indexing bugs
    def indexing_test(self, buckets_train_vectorized, verbose=1):
        X = np.array(buckets_train_vectorized)
        annoy = AnnoyIndex(X[0]['vector'].shape[0])  # Length of item vector that will be indexed

        loop = total=len(X)
        if(verbose):
            loop = tqdm(total=len(X))
            loop.set_description("Indexing test in annoy")
        for index, row in enumerate(X):
            vector = row['vector']
            annoy.add_item(index, vector)
            if(verbose): loop.update(1)
        if(verbose): loop.close()
        annoy.build(10) # 10 trees
        return annoy
    
    ## Rank result
    def rank_result(self, test_vectorized, indices_test, distance_test, verbose=1):
        formated_rank = []
        loop = zip(indices_test, distance_test)
        if(verbose):
            loop = tqdm(zip(indices_test, distance_test))
            loop.set_description('Generating the rank')
        for row_index, row_sim in loop:
            row_index, row_sim = row_index[:25], row_sim[:25]
            formated_rank.append(",".join(["{}:{}".format(test_vectorized[index]['bug_id'], sim) 
                                        for index, sim in zip(row_index, row_sim)]))
        if(verbose): loop.close()
        return formated_rank

    ## Vectorizer the test
    def vectorizer_test(self, bug_set, model, test, issues_by_buckets, method='keras', verbose=1):
        test_vectorized = []
        title_data, desc_data, info_data, title_desc_data = [], [], [], []
        loop = test
        if(verbose):
            loop = tqdm(test)
            loop.set_description('Vectorizing buckets')
        buckets = set()
        for row in loop: # retrieval.bugs_train
            query, ground_truth = row
            bugs = [query]
            bugs += ground_truth
            for bug_id in bugs:
                buckets.add(issues_by_buckets[bug_id])
        for bucket_id in buckets:
            bug = bug_set[bucket_id]
            if method == 'keras':
                title_data.append(bug['title_word'])
                desc_data.append(bug['description_word'])
                info_data.append(self.retrieval.get_info(bug))
            elif method == 'fasttext' or method == 'doc2vec':
                title_desc_data.append(bug['title'] + ' ' + bug['description'])
            test_vectorized.append({ 'bug_id' : bucket_id })
        if(verbose):
            loop.close()
        # Get embedding of all buckets
        if method == 'keras':
            embed_test = model.predict([ np.array(title_data), np.array(desc_data), np.array(info_data) ])
        elif method == 'fasttext':
            embed_test = [ model.get_sentence_vector(row) for row in title_desc_data ]
        elif method == 'doc2vec':
            embed_test = [ model.infer_vector(row.split(' ')) for row in title_desc_data ]
        # Fill the buckets array
        for index, vector in enumerate(embed_test):
            test_vectorized[index]['vector'] = vector
        
        return test_vectorized

    def vectorize_queries(self, bug_set, model, test, issues_by_buckets, method='keras', verbose=1):
        queries_test_vectorized = []
        title_data, desc_data, info_data, title_desc_data = [], [], [], []
        loop = test
        if(verbose):
            loop = tqdm(test)
        for row in loop:
            test_bug_id, ground_truth = row
            if issues_by_buckets[test_bug_id] == test_bug_id: # if the bug is the master
                test_bug_id = np.random.choice(ground_truth, 1)[0]
            queries = set()
            queries.add(test_bug_id)
            if test_bug_id in ground_truth:
                ground_truth = list(set(ground_truth) - set([test_bug_id])) # Remove the same bug random choice to change the master
            if len(ground_truth) > 0:
                for bug in ground_truth:
                    if issues_by_buckets[bug] != bug: # if the bug is the master
                        queries.add(bug)
                    
            for bug_id in queries:
                bug = bug_set[bug_id]
                if method == 'keras':
                    title_data.append(bug['title_word'])
                    desc_data.append(bug['description_word'])
                    info_data.append(self.retrieval.get_info(bug))
                elif method == 'fasttext' or method == 'doc2vec':
                    title_desc_data.append(bug['title'] + ' ' + bug['description'])
                
                queries_test_vectorized.append({ 'bug_id' : bug_id, 'ground_truth': issues_by_buckets[bug_id] })

        # Get embedding of all buckets
        if method == 'keras':
            embed_queries = model.predict([ np.array(title_data), np.array(desc_data), np.array(info_data) ])
        elif method == 'fasttext':
            embed_queries = [ model.get_sentence_vector(text) for text in title_desc_data ]
        elif method == 'doc2vec':
            embed_queries = [ model.infer_vector(text.split(' ')) for text in title_desc_data ]
        # Fill the queries array    
        for index, vector in enumerate(embed_queries):
            queries_test_vectorized[index]['vector'] = vector
        
        return queries_test_vectorized

    # Generating the rank result
    def formating_rank(self, X_test, verbose=1):
        rank_queries = []
        loop = enumerate(X_test)
        if(verbose):
            loop = tqdm(enumerate(X_test))
            loop.set_description('Generating the queries from rank')
        for index, row in loop:
            dup_a, ground_truth = row['bug_id'], row['ground_truth']
            rank_queries.append("{}:{}".format(dup_a, ground_truth))
        if(verbose): loop.close()
        return rank_queries
    
    def export_rank(self, rank_queries, formated_rank, verbose=1):
        exported_rank = []
        loop = len(rank_queries)
        if(verbose):
            loop = tqdm(total=len(rank_queries))
            loop.set_description('Exporting the rank')
        for query, rank in zip(rank_queries, formated_rank):
            exported_rank.append("{}|{}".format(query, rank))
            if(verbose): loop.update(1)
        if(verbose): loop.close()
        return exported_rank
    
    def evaluate_validation_test(self, retrieval, verbose, loaded_model, issues_by_buckets, method='keras'):
        # Load test set
        test = self.retrieval.test
        bug_set = self.retrieval.baseline.get_bug_set()
        
        # Get model
        model = self.get_model_vectorizer(loaded_model=loaded_model)
        
        # Test 
        test_vectorized = self.vectorizer_test(bug_set, model, test, issues_by_buckets, method, verbose)
        queries_test_vectorized = self.vectorize_queries(bug_set, model, test, issues_by_buckets, method, verbose)
        annoy = self.indexing_test(test_vectorized, verbose)
        X_test, distance_test, indices_test = self.indexing_query(annoy, queries_test_vectorized, verbose)
        formated_rank = self.rank_result(test_vectorized, indices_test, distance_test, verbose)
        rank_queries = self.formating_rank(X_test, verbose)
        exported_rank = self.export_rank(rank_queries, formated_rank, verbose)
        recall = self.evaluation.evaluate(exported_rank)['5 - recall_at_25']
        
        # recall@25, loss, cosine_positive, cosine_negative
        return recall, exported_rank
        #return report['5 - recall_at_25'], evaluation_test_batch[0], evaluation_test_batch[1], evaluation_test_batch[2]

    def evaluation(self, evaluation):
        self.evaluation = evaluation

    def save_model(self, model, name, verbose=0):
        m_dir = os.path.join('modelos')
        if not os.path.exists(m_dir):
            os.mkdir(m_dir)
        export = os.path.join(m_dir, "model_{}.h5".format(name))
        model.save(export)
        if(verbose):
            print("Saved model '{}' to disk".format(export))