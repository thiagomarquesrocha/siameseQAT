import os
import _pickle as pickle
import numpy as np
from tqdm import tqdm

class TrainingData:

    def __init__(self):
        pass

    def load_object(self, path):
        with open(os.path.join(DIR, '{}.pkl'.format(path)), 'rb') as f:
            return pickle.load(f)
    
    def save_object(self, DIR, path, obj):
        with open(os.path.join(DIR, '{}.pkl'.format(path)), 'wb') as f:
            pickle.dump(obj, f)

    # Load buckets preprocessed from analysing_buckets.ipynb
    def load_buckets(self, DIR, DOMAIN):
        with open(os.path.join(DIR, DOMAIN + '_buckets.pkl'), 'rb') as f:
            self.buckets = pickle.load(f)

    def get_info_dict(self, DIR, DOMAIN):

        if DOMAIN != 'firefox':
            self.info_dict = {
                'bug_severity' : self.get_feature_size(DIR, 'bug_severity'),
                'product' : self.get_feature_size(DIR, 'product'),
                'bug_status' : self.get_feature_size(DIR, 'bug_status'),
                'component' : self.get_feature_size(DIR, 'component'),
                'priority' : self.get_feature_size(DIR, 'priority'),
                'version' : self.get_feature_size(DIR, 'version')
            }
        else:
            self.info_dict = {
                'bug_status' : self.get_feature_size(DIR, 'bug_status'),
                'component' : self.get_feature_size(DIR, 'component'),
                'priority' : self.get_feature_size(DIR, 'priority'),
                'version' : self.get_feature_size(DIR, 'version')
            }
        self.categorical_size = self.get_categorical_size()

    def get_feature_size(self, DIR, name):
        with open(os.path.join(DIR, '{}.dic'.format(name)), 'rb') as f:
            features = str(f.read()).split('\\n')[:-1]
        return len(features)

    def get_categorical_size(self):
        return sum([total for total in self.info_dict.values()])

    def load_bugs(self, DIR, method, TOKEN_END, MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D):   
        removed = []
        self.bug_set = {}
        title_padding, desc_padding = [], []
        for bug_id in tqdm(self.bug_ids):
            try:
                bug = pickle.load(open(os.path.join(DIR, 'bugs', '{}.pkl'.format(bug_id)), 'rb'))
                title_padding.append(bug['title_token'][:MAX_SEQUENCE_LENGTH_T])
                desc_padding.append(bug['description_token'][:MAX_SEQUENCE_LENGTH_D])
                self.bug_set[bug_id] = bug
                #break
            except:
                removed.append(bug_id)
        
        # Padding
        title_padding = self.data_padding(TOKEN_END, title_padding, MAX_SEQUENCE_LENGTH_T, method)
        desc_padding = self.data_padding(TOKEN_END, desc_padding, MAX_SEQUENCE_LENGTH_D, method)
        
        for bug_id, bug_title, bug_desc in tqdm(zip(self.bug_ids, title_padding, desc_padding)):
            bug = self.bug_set[bug_id]
            bug['title'] = bug['title']
            bug['description'] = bug['description']
            bug['title_token'] = bug_title
            bug['description_token'] = bug_desc
            bug['textual_token'] = np.concatenate([bug_title, bug_desc], -1)
        
        if len(removed) > 0:
            for x in removed:
                self.bug_ids.remove(x)
            self.removed = removed
            print("{} were removed. To see the list call self.removed".format(len(removed)))

    def prepare_dataset(self, 
                        DIR, issues_by_buckets, 
                        path_train='train_chronological', 
                            path_test='test_chronological'):
        if not self.bug_set or len(self.bug_set) == 0:
            raise Exception('self.bug_set not initialized')

        try:
            self.train_data = self.load_object('train_data')
            self.dup_sets_train = self.load_object('dup_sets_train')
            self.test_data = self.load_object('test_data')
            self.dup_sets_test = self.load_object('dup_sets_test')
            self.bug_ids = self.load_object('bug_ids')
        except:
            self.train_data, self.dup_sets_train = TrainingData.read_train_data(issues_by_buckets, DIR, list(self.bug_set), path_train)
            self.test_data, self.dup_sets_test = TrainingData.read_test_data(DIR, list(self.bug_set), issues_by_buckets, path_test)
            self.bug_ids = TrainingData.read_bug_ids(DIR)
            self.save_object(DIR, 'train_data', self.train_data)
            self.save_object(DIR, 'dup_sets_train', self.dup_sets_train)
            self.save_object(DIR, 'test_data', self.test_data)
            self.save_object(DIR, 'dup_sets_test', self.dup_sets_test)
            self.save_object(DIR, 'bug_ids', self.bug_ids)

    def get_buckets_for_bugs(self):
        issues_by_buckets = {}
        for bucket in tqdm(self.buckets):
            issues_by_buckets[bucket] = bucket
            for issue in np.array(self.buckets[bucket]).tolist():
                issues_by_buckets[issue] = bucket
        return issues_by_buckets

    def prepare_buckets_for_bugs(self):
        self.issues_by_buckets = self.get_buckets_for_bugs()

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

    def load_train_ids(self, train_data):
        self.bug_train_ids = self.get_train_ids(train_data)

    def load_test_ids(self, test_data):
        self.bug_test_ids = self.get_test_ids(test_data)

    @staticmethod
    def read_test_data(data, bug_set, issues_by_buckets, path_test):
        test_data = []
        bug_ids = set()
        data_dup_sets = {}
        bug_set = np.asarray(bug_set, int)
        with open(os.path.join(data, '{}.txt'.format(path_test)), 'r') as f:
            for line in f:
                bugs = np.asarray(line.strip().split(), int)
                bugs = [bug for bug in bugs if int(bug) in bug_set] 
                if len(bugs) < 2:
                    continue
                
                for i, bug_id in enumerate(bugs):
                    bucket = issues_by_buckets[int(bug_id)]
                    if bucket not in data_dup_sets:
                        data_dup_sets[bucket] = set()
                    data_dup_sets[bucket].add(int(bug_id))
                    bug_ids.add(int(bug_id))
                    for dup_id in bugs[i+1:]:
                        data_dup_sets[bucket].add(int(dup_id))
                        test_data.append([int(bug_id), int(dup_id)])
                        bug_ids.add(int(dup_id))
        return test_data, list(bug_ids)

    @staticmethod
    def read_train_data(issues_by_buckets, data, bug_set, path_train):
        data_pairs = []
        data_dup_sets = {}
        print('Reading train data')
        with open(os.path.join(data, '{}.txt'.format(path_train)), 'r') as f:
            for line in f:
                bug1, bug2 = line.strip().split()
                bug1 = int(bug1)
                bug2 = int(bug2)
                '''
                    Some bugs duplicates point to one master that
                    does not exist in the dataset like openoffice master=152778
                '''
                if bug1 not in bug_set or bug2 not in bug_set: 
                    continue
                data_pairs.append([bug1, bug2])
                bucket = issues_by_buckets[bug1]
                if bucket not in data_dup_sets.keys():
                    data_dup_sets[bucket] = set()
                data_dup_sets[bucket].add(bug1)
                data_dup_sets[bucket].add(bug2)
        return data_pairs, data_dup_sets

    @staticmethod
    def read_bug_ids(data):
        bug_ids = []
        print('Reading bug ids')
        with open(os.path.join(data, 'bug_ids.txt'), 'r') as f:
            for line in f:
                bug_ids.append(int(line.strip()))
        return bug_ids

    def load_bug_ids(self, data):
        self.bug_ids = self.read_bug_ids(data)

    def data_padding(self, token_end, data, max_seq_length, method):
        seq_lengths = [len(seq) for seq in data]
        seq_lengths.append(6)
        #max_seq_length = min(max(seq_lengths), max_seq_length)
        padded_data = np.zeros(shape=[len(data), max_seq_length])
        for i, seq in enumerate(data):
            seq = seq[:max_seq_length]
            end_sent = -1
            for j, token in enumerate(seq):
                if(int(token) == token_end):
                    token = 0
                padded_data[i, j] = int(token)
            if method == 'bert':
                padded_data[i] = np.concatenate([padded_data[i][:-1], [token_end]])
        return padded_data.astype(np.int)