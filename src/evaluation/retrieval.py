from src.evaluation.recall import Recall
from src.utils.util import Util
from tqdm import tqdm
from annoy import AnnoyIndex
import numpy as np

class Retrieval:

    def __init__(self, DOMAIN, info_dict, verbose=True):
        self.DOMAIN = DOMAIN
        self.info_dict = info_dict
        self.verbose = verbose
        self.recall = Recall(verbose)

    def evaluate(self, buckets, test, bug_set, model, issues_by_buckets, bug_train_ids, method='keras', only_buckets=False):
        
        # Test 
        test_vectorized = self.vectorizer_test(buckets, bug_set, model, test, issues_by_buckets, method, only_buckets=only_buckets)
        queries_test_vectorized = self.vectorize_queries(buckets, bug_set, model, test, issues_by_buckets, bug_train_ids, method, only_buckets=only_buckets)
        annoy = self.indexing_test(test_vectorized)
        X_test, distance_test, indices_test = self.indexing_query(annoy, queries_test_vectorized)
        formated_rank = self.rank_result(X_test, test_vectorized, indices_test, distance_test)
        rank_queries = self.formating_rank(X_test)
        exported_rank = self.export_rank(rank_queries, formated_rank)
        recall = self.recall.evaluate(exported_rank)['5 - recall_at_25']
        
        # recall@25, loss, cosine_positive, cosine_negative
        return recall, exported_rank, [test_vectorized, queries_test_vectorized, annoy, X_test, distance_test, indices_test]
        #return report['5 - recall_at_25'], evaluation_test_batch[0], evaluation_test_batch[1], evaluation_test_batch[2]

    #### Getting the list of candidates
    def indexing_query(self, annoy, queries_test_vectorized):
        X_test = queries_test_vectorized
        distance_test, indices_test = [], []
        loop = enumerate(X_test)
        if(self.verbose):
            loop = tqdm(enumerate(X_test))
            loop.set_description('Getting the list of candidates from queries')
        for index, row in loop:
            vector = row['vector']
            rank, dist = annoy.get_nns_by_vector(vector, 30, include_distances=True)
            indices_test.append(rank)
            max_dist = np.amax(dist)
            max_dist = max_dist if(max_dist > 1) else 1
            distance_test.append(max_dist - np.array(dist)) # normalize the similarity between 0 and 1
        if(self.verbose): loop.close()
        return X_test, distance_test, indices_test

    def export_rank(self, rank_queries, formated_rank):
        exported_rank = []
        loop = len(rank_queries)
        if(self.verbose):
            loop = tqdm(total=len(rank_queries))
            loop.set_description('Exporting the rank')
        for query, rank in zip(rank_queries, formated_rank):
            exported_rank.append("{}|{}".format(query, rank))
            if(self.verbose): loop.update(1)
        if(self.verbose): loop.close()
        return exported_rank

    # Generating the rank result
    def formating_rank(self, X_test):
        rank_queries = []
        loop = enumerate(X_test)
        if(self.verbose):
            loop = tqdm(enumerate(X_test))
            loop.set_description('Generating the queries from rank')
        for index, row in loop:
            dup_a, ground_truth = row['bug_id'], row['ground_truth']
            rank_queries.append("{}:{}".format(dup_a, ",".join(np.asarray(ground_truth, str))))
        if(self.verbose): loop.close()
        return rank_queries

    ## Rank result
    def rank_result(self, X_test, test_vectorized, indices_test, distance_test):
        formated_rank = []
        loop = zip(indices_test, distance_test, X_test)
        if(self.verbose):
            loop = tqdm(zip(indices_test, distance_test, X_test))
            loop.set_description('Generating the rank')
        for row_index, row_sim, row_query in loop:
            row_index, row_sim = row_index[:30], row_sim[:30]
            formated_rank.append(",".join(["{}:{}".format(test_vectorized[index]['bug_id'], sim) 
                                        for index, sim in zip(row_index, row_sim) 
                                               if row_query['bug_id'] != test_vectorized[index]['bug_id']
                                          ]))
        if(self.verbose): loop.close()
        return formated_rank

    # Indexing bugs
    def indexing_test(self, bugs_test):
        X = np.array(bugs_test)
        annoy = AnnoyIndex(X[0]['vector'].shape[0])  # Length of item vector that will be indexed

        loop = total=len(X)
        if(self.verbose):
            loop = tqdm(total=len(X))
            loop.set_description("Indexing test in annoy")
        for index, row in enumerate(X):
            vector = row['vector']
            annoy.add_item(index, vector)
            if(self.verbose): loop.update(1)
        if(self.verbose): loop.close()
        annoy.build(10) # 10 trees
        return annoy

    ## Vectorizer the test
    def vectorizer_test(self, buckets, bug_set, model, test, issues_by_buckets, method='keras', only_buckets=False):
        if self.info_dict == None:
            raise Exception("info_dict was not initialized")
        
        test_vectorized = []
        title_data, desc_data, info_data, topic_data, title_desc_data = [], [], [], [], []
        loop = test
        if(self.verbose):
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
                bugs = buckets[issues_by_buckets[query]]
                for bug_id in bugs:
                    tests.add(bug_id)

        for bug_id in tests:
            if bug_id not in bug_set: # Firefox does not exit bug 131106
                continue
            bug = bug_set[bug_id]
            self.get_data_input(bug, method, title_data, desc_data, info_data, topic_data)

            test_vectorized.append({ 'bug_id' : bug_id })
        if(self.verbose):
            loop.close()
        # Get embedding of all buckets
        embed_test = self.get_embed(method, model, title_data, desc_data, info_data, topic_data)
        
        # Fill the buckets array
        for index, vector in enumerate(embed_test):
            test_vectorized[index]['vector'] = vector

        return test_vectorized

    def vectorize_queries(self, buckets, bug_set, model, test, issues_by_buckets, bug_train_ids, method='keras', only_buckets=False):
        queries_test_vectorized = []
        title_data, desc_data, info_data, topic_data, title_desc_data = [], [], [], [], []
        
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
                queries.add(test_bug_id)
                queries.add(ground_truth)
        
        loop = queries
        if(self.verbose):
            loop = tqdm(queries)
        
        for test_bug_id in loop:
            
            if only_buckets:
                ground_truth_fix = [issues_by_buckets[test_bug_id]]
            else:
                ground_truth_fix = list(buckets[issues_by_buckets[test_bug_id]])
                ground_truth_fix.remove(test_bug_id)
            
            if test_bug_id not in bug_set: # Firefox does not exit bug 131106
                continue
            bug = bug_set[test_bug_id]
            self.get_data_input(bug, method, title_data, desc_data, info_data, topic_data)
            queries_test_vectorized.append({ 'bug_id' : test_bug_id, 'ground_truth': ground_truth_fix })

        # Get embedding of all buckets
        embed_queries = self.get_embed(method, model, title_data, desc_data, info_data, topic_data)
        
        # Fill the queries array    
        for index, vector in enumerate(embed_queries):
            queries_test_vectorized[index]['vector'] = vector

        return queries_test_vectorized

    def get_attention_mask(self, arr):
        return np.array(np.array(arr) > 0.0, int)

    def get_embed(self, method, model, title_data, desc_data, info_data, topic_data):
        if method == 'keras':
            embed = model.predict([ np.array(info_data), np.array(desc_data), np.array(title_data) ])
        elif method == 'bert':
            embed = model.predict([ np.array(info_data), self.get_attention_mask(desc_data), np.array(desc_data), self.get_attention_mask(title_data), np.array(title_data) ])
        elif method == 'bert-topic':
            embed = model.predict([ np.array(info_data), self.get_attention_mask(desc_data), np.array(desc_data), np.array(title_data), self.get_attention_mask(title_data), np.array(topic_data) ])
        elif method == 'dwen':
            embed = model.predict([ np.array(desc_data), np.array(title_data) ])
        
        return embed
    
    def get_data_input(self, bug, method, title_data, desc_data, info_data, topic_data):
        if method == 'keras' or method == 'bert' or method == 'fake':
            title_data.append(bug['title_token'])
            desc_data.append(bug['description_token'])
            info_data.append(Util.get_info(bug, self.info_dict, self.DOMAIN))
        elif method == 'bert-topic':
            title_data.append(bug['title_token'])
            desc_data.append(bug['description_token'])
            topic_data.append(bug['topics'])
            info_data.append(Util.get_info(bug, self.info_dict, self.DOMAIN))
        elif method == 'dwen':
            title_data.append(bug['title_token'])
            desc_data.append(bug['description_token'])