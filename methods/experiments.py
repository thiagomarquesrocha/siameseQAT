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

    def prepare_dataset(self, issues_by_buckets, path_train='train', path_test='test'):
        self.baseline.prepare_dataset(issues_by_buckets, path_train, path_test)

    def set_retrieval(self, retrieval, baseline, DOMAIN):
        # Link references
        self.retrieval = retrieval
        retrieval.baseline = baseline

        # self.baseline.MAX_SEQUENCE_LENGTH_I = number_of_columns_info # Status, Severity, Version, Component, Module

        # Create the instance from baseline
        path_buckets = 'data/normalized/{}/{}.csv'.format(DOMAIN, DOMAIN)

        df = pd.read_csv(path_buckets)

        # Load bug ids
        #retrieval.load_bugs(path, path_train)
        # Create the buckets
        retrieval.create_bucket(df)
    
    def create_queries(self):
        print("Reading queries from baseline.")
        self.retrieval.create_queries()

    def get_buckets_for_bugs(self):
        return self.retrieval.get_buckets_for_bugs()
    
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
            max_dist = np.amax(dist)
            max_dist = max_dist if(max_dist > 1) else 1
            distance_test.append(max_dist - np.array(dist)) # normalize the similarity between 0 and 1
        if(verbose): loop.close()
        return X_test, distance_test, indices_test
    
    # Indexing bugs
    def indexing_test(self, bugs_test, verbose=1):
        X = np.array(bugs_test)
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
    def rank_result(self, X_test, test_vectorized, indices_test, distance_test, verbose=1):
        formated_rank = []
        loop = zip(indices_test, distance_test, X_test)
        if(verbose):
            loop = tqdm(zip(indices_test, distance_test, X_test))
            loop.set_description('Generating the rank')
        for row_index, row_sim, row_query in loop:
            row_index, row_sim = row_index[:30], row_sim[:30]
            formated_rank.append(",".join(["{}:{}".format(test_vectorized[index]['bug_id'], sim) 
                                        for index, sim in zip(row_index, row_sim) 
                                               if row_query['bug_id'] != test_vectorized[index]['bug_id']
                                          ]))
        if(verbose): loop.close()
        return formated_rank

    ## Vectorizer the test
    def vectorizer_test(self, bug_set, model, test, issues_by_buckets, method='keras', verbose=1):
        test_vectorized = []
        title_data, desc_data, info_data, title_desc_data = [], [], [], []
        loop = test
        if(verbose):
            loop = tqdm(test)
            loop.set_description('Vectorizing bugs')
        
        tests = set()
        for row in loop: # retrieval.bugs_train
            query, ground_truth = row
            bugs = self.retrieval.buckets[issues_by_buckets[query]]
            for bug_id in bugs:
                tests.add(bug_id)

        for bug_id in tests:
            bug = bug_set[bug_id]
            if method == 'keras':
                title_data.append(bug['title_word'])
                desc_data.append(bug['description_word'])
                info_data.append(self.retrieval.get_info(bug))
            if method == 'bert':
                title_data.append(bug['title_word_bert'])
                desc_data.append(bug['description_word_bert'])
                info_data.append(self.retrieval.get_info(bug))
            elif method == 'dwen':
                title_data.append(bug['title_word'])
                desc_data.append(bug['description_word'])
            elif method == 'fasttext' or method == 'doc2vec':
                title_desc_data.append(bug['title'] + ' ' + bug['description'])
            test_vectorized.append({ 'bug_id' : bug_id })
        if(verbose):
            loop.close()
        # Get embedding of all buckets
        if method == 'keras':
            embed_test = model.predict([ np.array(title_data), np.array(desc_data), np.array(info_data) ])
        elif method == 'bert':
            embed_test = model.predict([ np.array(title_data), np.zeros_like(title_data), np.array(desc_data), np.zeros_like(desc_data), np.array(info_data) ])
        elif method == 'dwen':
            embed_test = model.predict([ np.array(title_data), np.array(desc_data) ])
        elif method == 'fasttext':
            embed_test = [ model.get_sentence_vector(row) for row in title_desc_data ]
        elif method == 'doc2vec':
            embed_test = [ model.infer_vector(row.split(' ')) for row in title_desc_data ]
        # Fill the buckets array
        for index, vector in enumerate(embed_test):
            test_vectorized[index]['vector'] = vector

        return test_vectorized

    def vectorize_queries(self, bug_set, model, test, issues_by_buckets, bug_train_ids, method='keras', verbose=1):
        queries_test_vectorized = []
        title_data, desc_data, info_data, title_desc_data = [], [], [], []
        
        # Transform all duplicates in queries
        queries = []
        for row in test:
            test_bug_id, ground_truth = row
            if test_bug_id not in bug_train_ids:
                queries.append(test_bug_id)
            for bug_id in ground_truth:
                if bug_id not in bug_train_ids:
                    queries.append(bug_id)
        
        loop = queries
        if(verbose):
            loop = tqdm(queries)
        
        for test_bug_id in loop:
            
            ground_truth_fix = list(self.retrieval.buckets[issues_by_buckets[test_bug_id]])
            ground_truth_fix.remove(test_bug_id)

            bug = bug_set[test_bug_id]
            if method == 'keras':
                title_data.append(bug['title_word'])
                desc_data.append(bug['description_word'])
                info_data.append(self.retrieval.get_info(bug))
            if method == 'bert':
                title_data.append(bug['title_word_bert'])
                desc_data.append(bug['description_word_bert'])
                info_data.append(self.retrieval.get_info(bug))
            elif method == 'dwen':
                title_data.append(bug['title_word'])
                desc_data.append(bug['description_word'])
            elif method == 'fasttext' or method == 'doc2vec':
                title_desc_data.append(bug['title'] + ' ' + bug['description'])

            queries_test_vectorized.append({ 'bug_id' : test_bug_id, 'ground_truth': ground_truth_fix })

        # Get embedding of all buckets
        if method == 'keras':
            embed_queries = model.predict([ np.array(title_data), np.array(desc_data), np.array(info_data) ])
        elif method == 'bert':
            embed_queries = model.predict([ np.array(title_data), np.zeros_like(title_data), np.array(desc_data), np.zeros_like(desc_data), np.array(info_data) ])
        elif method == 'dwen':
            embed_queries = model.predict([ np.array(title_data), np.array(desc_data) ])
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
            rank_queries.append("{}:{}".format(dup_a, ",".join(np.asarray(ground_truth, str))))
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
    
    def evaluate_validation_test(self, retrieval, verbose, loaded_model, issues_by_buckets, bug_train_ids, method='keras'):
        # Load test set
        test = self.retrieval.test
        bug_set = self.retrieval.baseline.get_bug_set()
        
        # Get model
        model = self.get_model_vectorizer(loaded_model=loaded_model)
        
        # Test 
        test_vectorized = self.vectorizer_test(bug_set, model, test, issues_by_buckets, method, verbose)
        queries_test_vectorized = self.vectorize_queries(bug_set, model, test, issues_by_buckets, bug_train_ids, method, verbose)
        annoy = self.indexing_test(test_vectorized, verbose)
        X_test, distance_test, indices_test = self.indexing_query(annoy, queries_test_vectorized, verbose)
        formated_rank = self.rank_result(X_test, test_vectorized, indices_test, distance_test, verbose)
        rank_queries = self.formating_rank(X_test, verbose)
        exported_rank = self.export_rank(rank_queries, formated_rank, verbose)
        recall = self.evaluation.evaluate(exported_rank)['5 - recall_at_25']
        
        # recall@25, loss, cosine_positive, cosine_negative
        return recall, exported_rank, [test_vectorized, queries_test_vectorized, annoy, X_test, distance_test, indices_test]
        #return report['5 - recall_at_25'], evaluation_test_batch[0], evaluation_test_batch[1], evaluation_test_batch[2]

    def set_evaluation(self, evaluation):
        self.evaluation = evaluation

    def save_model(self, model, name, verbose=0):
        m_dir = os.path.join('modelos')
        if not os.path.exists(m_dir):
            os.mkdir(m_dir)
        export = os.path.join(m_dir, "model_{}.h5".format(name))
        model.save(export)
        if(verbose):
            print("Saved model '{}' to disk".format(export))

    def get_train_ids(self, train_data):
        bug_train_ids = []
        for pair in train_data:
            bug_train_ids.append(pair[0])
            bug_train_ids.append(pair[1])
        return bug_train_ids

    def get_test_ids(self, test_data):
        bug_test_ids = []
        for pair in test_data:
            bug_test_ids.append(pair[0])
            bug_test_ids.append(pair[1])
        return bug_test_ids

    def read_test_data_classification(self, data, bug_set, bug_train_ids, path='test'):
        data_dup_sets = {}
        test_data = []
        print('Reading test data for classification')
        with open(os.path.join(data, '{}.txt'.format(path)), 'r') as f:
            for line in f:
                bugs = line.strip().split()
                '''
                    Some bugs duplicates point to one master that
                    does not exist in the dataset like openoffice master=152778
                '''
                bugs = [bug for bug in bugs if int(bug) in bug_set and int(bug) not in bug_train_ids]
                if len(bugs) < 2:
                    continue
                query = int(bugs[0])
                dups = bugs[:1]
                if query not in data_dup_sets:
                    data_dup_sets[query] = set()
                for bug in dups:
                    bug = int(bug)
                    data_dup_sets[query].add(bug)
                    test_data.append([query, bug])
        return test_data, data_dup_sets