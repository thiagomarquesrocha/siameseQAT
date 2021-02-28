import random
import numpy as np
from deep_learning.training.training_data import TrainingData

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

    def to_one_hot(self, idx, size):
        one_hot = np.zeros(size)
        one_hot[int(float(idx))] = 1
        return one_hot

    def get_test_ids(self, test_data):
        bug_test_ids = []
        for pair in test_data:
            bug_test_ids.append(pair[0])
            bug_test_ids.append(pair[1])
        return bug_test_ids

    def read_batch_bugs(self, batch, bug, index=-1, title_ids=None, description_ids=None):
        if self.DOMAIN != 'firefox':
            info = np.concatenate((
                self.to_one_hot(bug['bug_severity'], self.info_dict['bug_severity']),
                self.to_one_hot(bug['bug_status'], self.info_dict['bug_status']),
                self.to_one_hot(bug['component'], self.info_dict['component']),
                self.to_one_hot(bug['priority'], self.info_dict['priority']),
                self.to_one_hot(bug['product'], self.info_dict['product']),
                self.to_one_hot(bug['version'], self.info_dict['version']))
            )
        else:
            info = np.concatenate((
                self.to_one_hot(bug['bug_status'], self.info_dict['bug_status']),
                self.to_one_hot(bug['component'], self.info_dict['component']),
                self.to_one_hot(bug['priority'], self.info_dict['priority']),
                self.to_one_hot(bug['version'], self.info_dict['version']))
            )
        #info.append(info_)
        if('topics' in bug and 'topics' in batch):
            batch['topics'].append(bug['topics'])
        batch['info'].append(info)
        batch['title'].append(bug['title_token'])
        batch['desc'].append(bug['description_token'])
        if(index != -1):
            title_ids[index] = [int(v > 0) for v in bug['title_token']]
            description_ids[index] = [int(v > 0) for v in bug['description_token']]

    def batch_iterator(self, buckets, data, bug_ids, batch_size, issues_by_buckets):
    
        random.shuffle(data)

        batch_features = {'title' : [], 'desc' : [], 'info' : [], 'topics' : []}
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
                bug_anchor = self.bug_set[anchor]
                bug_pos = self.bug_set[pos]
                if neg not in self.bug_set:
                    continue
                batch_bugs.append(neg)
                batch_bugs_neg.append(neg)
                bug_neg = self.bug_set[neg]
                break
            
            # triplet bug and master
            batch_triplets.append([anchor, pos, neg])
        
        random.shuffle(batch_bugs)
        title_ids = np.full((len(batch_bugs), self.MAX_SEQUENCE_LENGTH_T), 0)
        description_ids = np.full((len(batch_bugs), self.MAX_SEQUENCE_LENGTH_D), 0)
        for i, bug_id in enumerate(batch_bugs):
            bug = self.bug_set[bug_id]
            self.read_batch_bugs(batch_features, bug, index=i, 
                                    title_ids=title_ids, 
                                        description_ids=description_ids)

        batch_features['title'] = { 'token' : np.array(batch_features['title']), 'segment' : title_ids }
        batch_features['desc'] = { 'token' : np.array(batch_features['desc']), 'segment' : description_ids }
        batch_features['info'] = np.array(batch_features['info'])
        batch_features['topics'] = np.array(batch_features['topics'])
        
        sim = np.asarray([issues_by_buckets[bug_id] for bug_id in batch_bugs])

        input_sample = {}

        input_sample = { 'title' : batch_features['title'], 
                            'description' : batch_features['desc'], 
                                'info' : batch_features['info'],
                                'topics' : batch_features['topics'] }

        return batch_triplets, input_sample, sim #sim

    def run(self):
        data = self.get_data()
        data.load_buckets(self.DIR, self.DOMAIN)
        data.load_bug_ids(self.DIR)
        data.load_bugs(self.DIR, self.PREPROCESSING, self.TOKEN_END, 
                        self.MAX_SEQUENCE_LENGTH_T, self.MAX_SEQUENCE_LENGTH_D)
        data.prepare_buckets_for_bugs()
        data.prepare_dataset(self.DIR, data.issues_by_buckets)
        data.load_train_ids(data.train_data)

    def get_data(self):
        return self.training_data