from annoy import AnnoyIndex
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import os
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import _pickle as pickle

class Experiment:

    def __init__(self, baseline=None, evaluation=None):
        self.baseline = baseline
        self.evaluation = evaluation
    
    def load_ids(self):
        self.baseline.load_ids(self.baseline.DIR)
    
    def load_bugs(self, method='keras'):
        self.baseline.load_bugs(method=method)

    def batch_iterator(self, model, data, dup_sets, bug_ids, batch_size, n_neg, issues_by_buckets, TRIPLET_HARD=False, FLOATING_PADDING=False):
        return self.baseline.batch_iterator(self.retrieval, model, data, dup_sets, bug_ids, batch_size, n_neg, issues_by_buckets, TRIPLET_HARD=TRIPLET_HARD, FLOATING_PADDING=FLOATING_PADDING)

    def batch_classification_test(self, path, BERT=True):
        encoder = LabelEncoder()

        batch_1, batch_2 = {'title' : [], 'desc' : [], 'info' : []}, \
                                            {'title' : [], 'desc' : [], 'info' : []}

        batch_triplets, sim = [], []
        
        batch_1_t, batch_1_d = [], []
        batch_2_t, batch_2_d = [], []
        
        def token(title_token_ids, desc_token_ids, bug):
            title_token_ids.append([int(v > 0) for v in bug['title_token']])
            desc_token_ids.append([int(v > 0) for v in bug['description_token']])

        with open(path, 'r') as file_in:
            for row in file_in:
                test = row.split()
                bug1 = int(test[0])
                bug2 = int(test[1])
                label = int(test[2])
                bug1 = self.baseline.bug_set[bug1]
                bug2 = self.baseline.bug_set[bug2]
                self.baseline.read_batch_bugs(batch_1, bug1)
                self.baseline.read_batch_bugs(batch_2, bug2)
                token(batch_1_t, batch_1_d, bug1)
                token(batch_2_t, batch_2_d, bug2)
                sim.append(label)

        sim = encoder.fit_transform(sim)
        sim = to_categorical(sim)

        title_a = np.array(batch_1['title'])
        title_b = np.array(batch_2['title'])
        desc_a = np.array(batch_1['desc'])
        desc_b = np.array(batch_2['desc'])
        info_a = np.array(batch_1['info'])
        info_b = np.array(batch_2['info'])
        
        if(BERT):
            batch_1_t = np.asarray(batch_1_t)
            batch_2_t = np.asarray(batch_2_t)
            batch_1_d = np.asarray(batch_1_d)
            batch_2_d = np.asarray(batch_2_d)
            return title_a, batch_1_t, title_b, batch_2_t, desc_a, batch_1_d, desc_b, batch_2_d, info_a, info_b, sim
        return title_a, title_b, desc_a, desc_b, info_a, info_b, sim

    def get_centroid(self, dups, model, method):
        baseline = self.baseline
        retrieval = self.retrieval
        title_data, desc_data, info_data = [], [], []
        dups = [bug for bug in dups if bug in baseline.bug_set]
        for bug_id in dups:
            bug = baseline.bug_set[bug_id]
            title_data.append(bug['title_token'])
            desc_data.append(bug['description_token'])
            info_data.append(retrieval.get_info(bug))
        if method == 'bert':
            embeds = model.predict([ np.array(title_data), np.full((len(title_data), len(title_data[0])), 1), np.array(desc_data), np.full((len(desc_data), len(desc_data[0])), 1), np.array(info_data) ])
        elif method == 'keras':
            embeds = model.predict([ np.array(title_data), np.array(desc_data), np.array(info_data) ])
        kmeans = KMeans(n_clusters=1, random_state=0).fit(embeds)
        master_centroid = kmeans.cluster_centers_.tolist()[0]
        return { 'centroid_embed' : master_centroid }
    
    def batch_iterator_bert(self, model, data, dup_sets, bug_train_ids, batch_size, 
                                n_neg, issues_by_buckets, INCLUDE_MASTER=False, USE_CENTROID=False,
                                TRIPLET_HARD=False, FLOATING_PADDING=False, method='bert'):
        baseline = self.baseline
        retrieval = self.retrieval
        # global train_data
        # global self.dup_sets
        # global self.bug_ids
        # global self.bug_set

        random.shuffle(data)

        batch_input, batch_pos, batch_neg, master_batch_input, master_batch_neg = {'title' : [], 'desc' : [], 'info' : []}, \
                                                {'title' : [], 'desc' : [], 'info' : []}, \
                                                    {'title' : [], 'desc' : [], 'info' : []},\
                                                        {'title' : [], 'desc' : [], 'info' : [], 'centroid_embed': []}, \
                                                            {'title' : [], 'desc' : [], 'info' : [], 'centroid_embed': []}

        n_train = len(data)
        all_bugs = list(issues_by_buckets.keys())
        buckets = retrieval.buckets
        batch_triplets, batch_bugs_anchor, batch_bugs_pos, \
            batch_bugs_neg, batch_bugs = [], [], [], [], []

        for offset in range(batch_size):
            anchor, pos = data[offset][0], data[offset][1]
            batch_bugs_anchor.append(anchor)
            batch_bugs_pos.append(pos)
            batch_bugs.append(anchor)
            batch_bugs.append(pos)
        
        for anchor, pos in zip(batch_bugs_anchor, batch_bugs_pos):
            while True:
                if not TRIPLET_HARD:
                    neg = baseline.get_neg_bug(anchor, buckets[issues_by_buckets[anchor]], issues_by_buckets, all_bugs)
                else:
                    neg = baseline.get_neg_bug_semihard(self.retrieval, model, batch_bugs, anchor, pos, buckets[issues_by_buckets[anchor]], method=method)

                if neg not in baseline.bug_set \
                    or ((INCLUDE_MASTER or USE_CENTROID) and issues_by_buckets[neg] not in baseline.bug_set):
                    continue
                batch_bugs.append(neg)
                batch_bugs_neg.append(neg)
                break
        
        # Mask to BERT
        title_anchor_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_T), 0)
        description_anchor_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_D), 0)
        title_pos_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_T), 0)
        description_pos_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_D), 0)
        title_neg_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_T), 0)
        description_neg_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_D), 0)

        if INCLUDE_MASTER:
            title_master_pos_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_T), 0)
            description_master_pos_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_D), 0)
            title_master_neg_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_T), 0)
            description_master_neg_ids = np.full((batch_size, baseline.MAX_SEQUENCE_LENGTH_D), 0)

        for (i, anchor), pos, neg in zip(enumerate(batch_bugs_anchor), batch_bugs_pos, batch_bugs_neg):
            bug_anchor = baseline.bug_set[anchor]
            bug_pos = baseline.bug_set[pos]
            bug_neg = baseline.bug_set[neg]
            # master anchor and neg
            if INCLUDE_MASTER:
                master_anchor = baseline.bug_set[issues_by_buckets[anchor]]
                master_neg_id = issues_by_buckets[neg]
                master_neg = baseline.bug_set[master_neg_id]
            elif USE_CENTROID:
                master_anchor = self.get_centroid(retrieval.buckets[issues_by_buckets[anchor]], model, method=method)
                master_neg = self.get_centroid(retrieval.buckets[issues_by_buckets[neg]], model, method=method)
            
            baseline.read_batch_bugs(batch_input, bug_anchor, i, title_anchor_ids, description_anchor_ids)
            baseline.read_batch_bugs(batch_pos, bug_pos, i, title_pos_ids, description_pos_ids)
            baseline.read_batch_bugs(batch_neg, bug_neg, i, title_neg_ids, description_neg_ids)

            # check padding of desc field
            if(FLOATING_PADDING):
                self.baseline.apply_window_padding(bug_anchor, bug_pos)
                self.baseline.apply_window_padding(bug_anchor, bug_neg)
                self.baseline.apply_window_padding(bug_pos, bug_neg)

            # master anchor and neg
            if INCLUDE_MASTER:
                baseline.read_batch_bugs(master_batch_input, master_anchor, i, title_master_pos_ids, description_master_pos_ids)
                baseline.read_batch_bugs(master_batch_neg, master_neg, i, title_master_neg_ids, description_master_neg_ids)
                # quintet for bugs and masters
                batch_triplets.append([anchor, pos, neg, master_anchor, master_neg])
            elif USE_CENTROID:
                baseline.read_batch_bugs_centroid(master_batch_input, master_anchor)
                baseline.read_batch_bugs_centroid(master_batch_neg, master_neg)
                # quintet for bugs and masters
                batch_triplets.append([anchor, pos, neg, master_anchor, master_neg])
            else: # triplet for bugs
                batch_triplets.append([anchor, pos, neg])

        batch_input['title'] = { 'token' : np.array(batch_input['title']), 'segment' : title_anchor_ids }
        batch_input['desc'] = { 'token' : np.array(batch_input['desc']), 'segment' : description_anchor_ids }
        batch_input['info'] = np.array(batch_input['info'])
        batch_pos['title'] = { 'token' : np.array(batch_pos['title']), 'segment' : title_pos_ids }
        batch_pos['desc'] = { 'token' : np.array(batch_pos['desc']), 'segment' : description_pos_ids }
        batch_pos['info'] = np.array(batch_pos['info'])
        batch_neg['title'] = { 'token' : np.array(batch_neg['title']), 'segment' : title_neg_ids }
        batch_neg['desc'] = { 'token' : np.array(batch_neg['desc']), 'segment' : description_neg_ids }
        batch_neg['info'] = np.array(batch_neg['info'])
        
        # master
        if INCLUDE_MASTER:
            master_batch_input['title'] = { 'token' : np.array(master_batch_input['title']), 'segment' : title_master_pos_ids }
            master_batch_input['desc'] ={ 'token' : np.array(master_batch_input['desc']), 'segment' : description_master_pos_ids }
            master_batch_input['info'] = np.array(master_batch_input['info'])
            
            master_batch_neg['title'] = { 'token' : np.array(master_batch_neg['title']), 'segment' : title_master_neg_ids }
            master_batch_neg['desc'] = { 'token' : np.array(master_batch_neg['desc']), 'segment' : description_master_neg_ids }
            master_batch_neg['info'] = np.array(master_batch_neg['info'])
        elif USE_CENTROID:
            master_batch_input['centroid_embed'] = np.array(master_batch_input['centroid_embed'])
            master_batch_neg['centroid_embed'] = np.array(master_batch_neg['centroid_embed'])

        n_half = len(batch_triplets) // 2
        if n_half > 0:
            pos = np.full((1, n_half), 1)
            neg = np.full((1, n_half), 0)
            sim = np.concatenate([pos, neg], -1)[0]
        else:
            sim = np.array([np.random.choice([1, 0])])

        input_sample, input_pos, input_neg, master_input_sample, master_neg_sample = {}, {}, {}, {}, {}

        input_sample = { 'title' : batch_input['title'], 'description' : batch_input['desc'], 'info' : batch_input['info'] }
        input_pos = { 'title' : batch_pos['title'], 'description' : batch_pos['desc'], 'info': batch_pos['info'] }
        input_neg = { 'title' : batch_neg['title'], 'description' : batch_neg['desc'], 'info': batch_neg['info'] }
        # master
        if INCLUDE_MASTER: 
            master_input_sample = { 'title' : master_batch_input['title'], 'description' : master_batch_input['desc'], 
                                'info' : master_batch_input['info'] }
            master_neg_sample = { 'title' : master_batch_neg['title'], 'description' : master_batch_neg['desc'], 
                                'info' : master_batch_neg['info'] }
            return batch_triplets, input_sample, input_pos, input_neg, master_input_sample, master_neg_sample, sim #sim
        elif USE_CENTROID:
            master_input_sample = { 'centroid_embed': master_batch_input['centroid_embed'] }
            master_neg_sample = { 'centroid_embed' : master_batch_neg['centroid_embed'] }
            return batch_triplets, input_sample, input_pos, input_neg, master_input_sample, master_neg_sample, sim #sim
        else:
            return batch_triplets, input_sample, input_pos, input_neg, sim #sim

    def prepare_dataset(self, issues_by_buckets, path_train='train', path_test='test'):
        self.baseline.prepare_dataset(issues_by_buckets, path_train, path_test)

    def set_retrieval(self, retrieval, baseline, DOMAIN):
        # Link references
        self.retrieval = retrieval
        retrieval.baseline = baseline
        # Load buckets preprocessed from analysing_buckets.ipynb
        with open(os.path.join(baseline.DIR, DOMAIN + '_buckets.pkl'), 'rb') as f:
            self.retrieval.buckets = pickle.load(f)
    
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
    def vectorizer_test(self, bug_set, model, test, issues_by_buckets, method='keras', verbose=1, only_buckets=False):
        test_vectorized = []
        title_data, desc_data, info_data, title_desc_data = [], [], [], []
        loop = test
        if(verbose):
            loop = tqdm(test)
            loop.set_description('Vectorizing bugs')
        
        tests = set()
        for row in loop: # retrieval.bugs_train
            query, ground_truth = row
            if only_buckets: # only add buckets
                bugs = [query]
                bugs += ground_truth
                for bug_id in bugs: 
                    tests.add(issues_by_buckets[bug_id])
            else:
                bugs = self.retrieval.buckets[issues_by_buckets[query]]
                for bug_id in bugs:
                    tests.add(bug_id)

        for bug_id in tests:
            if bug_id not in bug_set: # Firefox does not exit bug 131106
                continue
            bug = bug_set[bug_id]
            if method == 'keras':
                title_data.append(bug['title_token'])
                desc_data.append(bug['description_token'])
                info_data.append(self.retrieval.get_info(bug))
            if method == 'bert':
                title_data.append(bug['title_token'])
                desc_data.append(bug['description_token'])
                info_data.append(self.retrieval.get_info(bug))
            elif method == 'dwen':
                title_data.append(bug['title_token'])
                desc_data.append(bug['description_token'])
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

    def vectorize_queries(self, bug_set, model, test, issues_by_buckets, bug_train_ids, method='keras', verbose=1, only_buckets=False):
        queries_test_vectorized = []
        title_data, desc_data, info_data, title_desc_data = [], [], [], []
        
        # Transform all duplicates in queries
        queries = set()
        for row in test:
            test_bug_id, ground_truth = row
            if only_buckets:
                if issues_by_buckets[test_bug_id] == test_bug_id: # if the bug is the master
                    test_bug_id = np.random.choice(ground_truth, 1)[0]
                queries.add(test_bug_id)
                if test_bug_id in ground_truth:
                    ground_truth = list(set(ground_truth) - set([test_bug_id])) # Remove the same bug random choice to change the master
                if len(ground_truth) > 0:
                    for bug in ground_truth:
                        if issues_by_buckets[bug] != bug: # if the bug is the master
                            queries.add(bug)
            else:
                # if test_bug_id not in bug_train_ids:
                queries.add(test_bug_id)
                for bug_id in ground_truth:
                    #if bug_id not in bug_train_ids:
                    queries.add(bug_id)
        
        loop = queries
        if(verbose):
            loop = tqdm(queries)
        
        for test_bug_id in loop:
            
            if only_buckets:
                ground_truth_fix = [issues_by_buckets[test_bug_id]]
            else:
                ground_truth_fix = list(self.retrieval.buckets[issues_by_buckets[test_bug_id]])
                ground_truth_fix.remove(test_bug_id)
            
            if test_bug_id not in bug_set: # Firefox does not exit bug 131106
                continue
            bug = bug_set[test_bug_id]
            if method == 'keras':
                title_data.append(bug['title_token'])
                desc_data.append(bug['description_token'])
                info_data.append(self.retrieval.get_info(bug))
            if method == 'bert':
                title_data.append(bug['title_token'])
                desc_data.append(bug['description_token'])
                info_data.append(self.retrieval.get_info(bug))
            elif method == 'dwen':
                title_data.append(bug['title_token'])
                desc_data.append(bug['description_token'])
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
    
    def evaluate_validation_test(self, retrieval, verbose, loaded_model, issues_by_buckets, bug_train_ids, method='keras', only_buckets=False):
        # Load test set
        test = self.retrieval.test
        bug_set = self.retrieval.baseline.get_bug_set()
        
        # Get model
        model = self.get_model_vectorizer(loaded_model=loaded_model)
        
        # Test 
        test_vectorized = self.vectorizer_test(bug_set, model, test, issues_by_buckets, method, verbose, only_buckets=only_buckets)
        queries_test_vectorized = self.vectorize_queries(bug_set, model, test, issues_by_buckets, bug_train_ids, method, verbose, only_buckets=only_buckets)
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
                bugs = [bug for bug in bugs if int(bug) in bug_set] # and int(bug) not in bug_train_ids
                if len(bugs) < 2:
                    continue
                query = int(bugs[0])
                dups = bugs[1:]
                if query not in data_dup_sets:
                    data_dup_sets[query] = set()
                for bug in dups:
                    bug = int(bug)
                    data_dup_sets[query].add(bug)
                    test_data.append([query, bug])
        return test_data, data_dup_sets