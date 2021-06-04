import random
import numpy as np
from src.utils.util import Util
from src.deep_learning.training.training_data import TrainingData

class TrainingPreparation():

    def __init__(self, DIR, DOMAIN, PREPROCESSING,
                    MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D, TOKEN_END):
        self.DIR = DIR
        self.DOMAIN = DOMAIN
        self.MAX_SEQUENCE_LENGTH_T = MAX_SEQUENCE_LENGTH_T
        self.MAX_SEQUENCE_LENGTH_D = MAX_SEQUENCE_LENGTH_D
        self.TOKEN_END = TOKEN_END
        self.PREPROCESSING = PREPROCESSING
        self.training_data = TrainingData()

    def get_neg_bug(self, invalid_bugs, bug_ids, issues_by_buckets, all_bugs):
        neg_bug = random.choice(all_bugs)
        bug_ids = list(bug_ids)
        try:
            while neg_bug in invalid_bugs or neg_bug not in issues_by_buckets:
                neg_bug = random.choice(bug_ids)
        except:
            invalid_bugs = [invalid_bugs]
            while neg_bug in invalid_bugs or neg_bug not in issues_by_buckets:
                neg_bug = random.choice(bug_ids)
        return neg_bug

    def read_batch_bugs(self, batch, bug, index=-1, title_ids=None, description_ids=None):
        info_dict = self.get_data().info_dict
        info = Util.get_info(bug, info_dict, self.DOMAIN)
        if('topics' in bug and 'topics' in batch):
            batch['topics'].append(bug['topics'])
        batch['info'].append(info)
        batch['title'].append(bug['title_token'])
        batch['desc'].append(bug['description_token'])
        if(index != -1):
            title_ids[index] = [int(v > 0) for v in bug['title_token']]
            description_ids[index] = [int(v > 0) for v in bug['description_token']]

    def batch_iterator(self, bug_set, buckets, data, bug_ids, batch_size, issues_by_buckets):
    
        random.shuffle(data)

        batch_features = {'title' : [], 'desc' : [], 'info' : [], 'topics' : []}
        batch_anchor_features = {'title' : [], 'desc' : [], 'info' : [], 'topics' : []}
        batch_pos_features = {'title' : [], 'desc' : [], 'info' : [], 'topics' : []}
        batch_neg_features = {'title' : [], 'desc' : [], 'info' : [], 'topics' : []}
        n_train = len(data)

        batch_triplets, batch_bugs_anchor, batch_bugs_pos, batch_bugs_neg, batch_bugs = [], [], [], [], []

        for offset in range(batch_size):
            anchor, pos = data[offset][0], data[offset][1]
            batch_bugs_anchor.append(anchor)
            batch_bugs_pos.append(pos)
            batch_bugs.append(anchor)
            batch_bugs.append(pos)

        for anchor, pos in zip(batch_bugs_anchor, batch_bugs_pos):
            while True:
                neg = self.get_neg_bug(anchor, 
                                    buckets[issues_by_buckets[anchor]], 
                                        issues_by_buckets, bug_ids)
                bug_anchor = bug_set[anchor]
                bug_pos = bug_set[pos]
                if neg not in bug_set:
                    continue
                batch_bugs.append(neg)
                batch_bugs_neg.append(neg)
                bug_neg = bug_set[neg]
                break
            
            # triplet bug and master
            batch_triplets.append([anchor, pos, neg])
        
        self.fill_batch_features(batch_bugs, batch_features, bug_set)
        self.fill_batch_features(batch_bugs_anchor, batch_anchor_features, bug_set)
        self.fill_batch_features(batch_bugs_pos, batch_pos_features, bug_set)
        self.fill_batch_features(batch_bugs_neg, batch_neg_features, bug_set)
        
        sim = np.asarray([issues_by_buckets[bug_id] for bug_id in batch_bugs])

        input_sample, input_anchor, input_pos, input_neg = {}, {}, {}, {}

        input_sample = self.create_batch_feature(batch_features)
        input_anchor = self.create_batch_feature(batch_anchor_features)
        input_pos = self.create_batch_feature(batch_pos_features)
        input_neg = self.create_batch_feature(batch_neg_features)

        return batch_triplets, input_sample, input_anchor, input_pos, input_neg, sim #sim

    def create_batch_feature(self, batch_features):
        return { 'title' : batch_features['title'], 
                            'description' : batch_features['desc'], 
                                'info' : batch_features['info'],
                                'topics' : batch_features['topics'] }

    def fill_batch_features(self, batch_bugs, batch_features, bug_set):
        title_ids = np.full((len(batch_bugs), self.MAX_SEQUENCE_LENGTH_T), 0)
        description_ids = np.full((len(batch_bugs), self.MAX_SEQUENCE_LENGTH_D), 0)
        for i, bug_id in enumerate(batch_bugs):
            bug = bug_set[bug_id]
            self.read_batch_bugs(batch_features, bug, index=i, 
                                    title_ids=title_ids, 
                                        description_ids=description_ids)

        batch_features['title'] = { 'token' : np.array(batch_features['title']), 'segment' : title_ids }
        batch_features['desc'] = { 'token' : np.array(batch_features['desc']), 'segment' : description_ids }
        batch_features['info'] = np.array(batch_features['info'])
        batch_features['topics'] = np.array(batch_features['topics'])

    def run(self):
        data = self.get_data()
        data.get_info_dict(self.DIR, self.DOMAIN)
        data.load_buckets(self.DIR, self.DOMAIN)
        data.load_bug_ids(self.DIR)
        data.load_bugs(self.DIR, self.PREPROCESSING, self.TOKEN_END, 
                        self.MAX_SEQUENCE_LENGTH_T, self.MAX_SEQUENCE_LENGTH_D)
        data.prepare_buckets_for_bugs()
        data.prepare_dataset(self.DIR, data.issues_by_buckets)
        data.load_train_ids(data.train_data)
        data.load_test_ids(data.test_data)

    def get_data(self):
        return self.training_data