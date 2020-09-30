from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import sklearn.metrics
import _pickle as pickle
from sklearn.manifold import TSNE
import random
import numpy as np
from tqdm import tqdm_notebook as tqdm

from tensorflow.python.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

from keras.layers import Conv1D, Input, Add, Activation, Dropout, Embedding, MaxPooling1D, GlobalMaxPool1D, Flatten, Dense, Concatenate, BatchNormalization
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.initializers import TruncatedNormal
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import optimizers
from keras import backend as K
import tensorflow as tf
import pandas as pd
from annoy import AnnoyIndex

class Baseline:

    def __init__(self, DOMAIN, DIR, dataset, MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D,
        TOKEN_BEGIN, TOKEN_END):
        self.sentence_dict = {}
        self.corpus = []
        self.bug_ids = []

        self.train_data = None
        self.test_data = None
        self.dup_sets_train = None
        self.dup_sets_test = None
        self.bug_set = {}

        self.DOMAIN = DOMAIN
        self.DIR = DIR
        self.GLOVE_DIR = ""
        self.MAX_SEQUENCE_LENGTH_T = MAX_SEQUENCE_LENGTH_T
        self.MAX_SEQUENCE_LENGTH_D = MAX_SEQUENCE_LENGTH_D
        self.TOKEN_BEGIN = TOKEN_BEGIN
        self.TOKEN_END = TOKEN_END
        self.get_info_dict(DIR)

    def get_feature_size(self, DIR, name):
        with open(os.path.join(DIR, '{}.dic'.format(name)), 'rb') as f:
            features = str(f.read()).split('\\n')[:-1]
        return len(features)

    def get_info_dict(self, DIR):
        # self.info_dict = {'bug_severity': 7, 'bug_status': 3, 'component': 323, 'priority': 5, 'product': 116, 'version': 197}
        
        if self.DOMAIN != 'firefox':
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

    def load_ids(self, DIR):
        self.bug_ids = self.read_bug_ids(DIR)

    @staticmethod
    def validation_accuracy_loss(history):
        acc=history.history['acc']
        val_acc=history.history['val_acc']
        loss=history.history['loss']
        val_loss=history.history['val_loss']

        plt.plot(acc, label='acc')
        plt.plot(val_acc, label='val_acc')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(loss, label='loss')
        plt.plot(val_loss, label='val_loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    @staticmethod
    def curve_roc_auc(model, x, y_valid):
        y_hat = model.predict(x)
        pct_auc = roc_auc_score(y_valid, y_hat) * 100
        #print('ROC/AUC: {:0.2f}'.format(pct_auc))

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_valid, y_hat)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def show_model_output(valid_a, valid_b, valid_sim, model, nb_examples = 3):
        #pv_a, pv_b, pv_sim = gen_random_batch(test_groups, nb_examples)
        pred_sim = model.predict([valid_a, valid_b])
        #     pred_sim = [1,1,1,1,1,1]
        for b_a, b_b, sim, pred in zip(valid_a, valid_b, valid_sim, pred_sim):
            key_a = ','.join(b_a.astype(str))
            key_b = ','.join(b_b.astype(str))
            print(sentence_dict[key_a])
            print(sentence_dict[key_b])
            print("similar=" + str(sim))
            print("prediction=" + str(pred[0]))
            print("########################")
        return valid_a, valid_b, valid_sim

    @staticmethod
    def load_model(DIR, name, dependences):
        m_dir = os.path.join(DIR, 'modelos')
        # load json and create model
        json_file = open(os.path.join(m_dir, "model_{}.json".format(name)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, dependences)
        # load weights into new model
        loaded_model.load_weights(os.path.join(m_dir, "model_{}.h5".format(name)))
        print("Loaded model from disk")
        return loaded_model

    @staticmethod
    def save_result(DIR, h, name):
        r_dir = os.path.join(DIR, 'resultados')
        if not os.path.exists(r_dir):
            os.mkdir(r_dir)
        with open(os.path.join(r_dir, name + '.pkl'), 'wb') as f:
            pickle.dump(h, f)

    @staticmethod    
    def load_result(DIR, name):
        with open(os.path.join(DIR, 'resultados', name + '.pkl'), 'r') as f:
            return pickle.load(f)

    ############ TSNE ###########
    @staticmethod
    def create_features(x_test_features):
        tsne_obj = TSNE(n_components=2,
                                init='pca',
                                random_state=101,
                                method='barnes_hut',
                                n_iter=500,
                                verbose=0)
        tsne_features = tsne_obj.fit_transform(x_test_features)
        return tsne_features

    @staticmethod
    def decode_to_categorical(datum):
        return np.argmax(datum)
    
    @staticmethod
    def plot_2d(test_labels, tsne_features):
        obj_categories = ['anchor', 'positive', 'negative']
        groups = [0, 1, 2]
        colors = plt.cm.rainbow(np.linspace(0, 1, 3))
        plt.figure(figsize=(10, 10))
        
        for c_group, (c_color, c_label) in enumerate(zip(colors, obj_categories)):
            plt.scatter(tsne_features[np.where(test_labels == c_group), 0],
                        tsne_features[np.where(test_labels == c_group), 1],
                        marker='o',
                        color=c_color,
                        linewidth='1',
                        alpha=0.8,
                        label=c_label)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('t-SNE on Testing Samples')
        plt.legend(loc='best')
        #plt.savefig('clothes-dist.png')
        plt.show(block=False)

    def display_embed_space(self, similarity_model, batch_size):
        valid_input_sample, valid_input_pos, valid_input_neg, valid_sim = self.batch_iterator(self.DIR, batch_size, 1)
        
        model_anchor  = similarity_model.get_layer('merge_features_in').output
        model_final = Model(inputs=similarity_model.input, outputs=model_anchor)
        x_test_features_anchor = model_final.predict([valid_input_sample['title'], valid_input_pos['title'], valid_input_neg['title'], 
                        valid_input_sample['description'], valid_input_pos['description'], valid_input_neg['description'],
                        valid_input_sample['info'], valid_input_pos['info'], valid_input_neg['info']], verbose = False, 
                                                batch_size=batch_size)

        model_pos  = similarity_model.get_layer('merge_features_pos').output
        model_final = Model(inputs=similarity_model.input, outputs=model_pos)
        x_test_features_pos = model_final.predict([valid_input_sample['title'], valid_input_pos['title'], valid_input_neg['title'], 
                        valid_input_sample['description'], valid_input_pos['description'], valid_input_neg['description'],
                        valid_input_sample['info'], valid_input_pos['info'], valid_input_neg['info']], verbose = False, 
                                                batch_size=batch_size)

        model_neg  = similarity_model.get_layer('merge_features_neg').output
        model_final = Model(inputs=similarity_model.input, outputs=model_neg)
        x_test_features_neg = model_final.predict([valid_input_sample['title'], valid_input_pos['title'], valid_input_neg['title'], 
                        valid_input_sample['description'], valid_input_pos['description'], valid_input_neg['description'],
                        valid_input_sample['info'], valid_input_pos['info'], valid_input_neg['info']], verbose = False, 
                                                batch_size=batch_size)
        
        #print("Shape", x_test_features_anchor.shape)
        
        x_test_features = np.concatenate([x_test_features_anchor, x_test_features_pos, x_test_features_neg], axis=0)
        
        #print("features", x_test_features.shape)
        
        anchor = np.full((1, batch_size), 0)
        pos = np.full((1, batch_size), 1)
        neg = np.full((1, batch_size), 2)
        valid_sim = np.concatenate([anchor, pos, neg], -1)[0]
        
        tsne_features = Baseline.create_features(x_test_features)

        Baseline.plot_2d(valid_sim, tsne_features)

    def load_bugs(self, method):   
        removed = []
        self.corpus = []
        self.sentence_dict = {}
        self.bug_set = {}
        title_padding, desc_padding = [], []
        for bug_id in tqdm(self.bug_ids):
            try:
                bug = pickle.load(open(os.path.join(self.DIR, 'bugs', '{}.pkl'.format(bug_id)), 'rb'))
                title_padding.append(bug['title_token'][:self.MAX_SEQUENCE_LENGTH_T])
                desc_padding.append(bug['description_token'][:self.MAX_SEQUENCE_LENGTH_D])
                self.bug_set[bug_id] = bug
                #break
            except:
                removed.append(bug_id)
        
        # Padding
        title_padding = self.data_padding(title_padding, self.MAX_SEQUENCE_LENGTH_T, method=method)
        desc_padding = self.data_padding(desc_padding, self.MAX_SEQUENCE_LENGTH_D, method=method)
        
        for bug_id, bug_title, bug_desc in tqdm(zip(self.bug_ids, title_padding, desc_padding)):
            bug = self.bug_set[bug_id]
            self.sentence_dict[",".join(np.array(bug_title, str))] = bug['title']
            self.sentence_dict[",".join(np.array(bug_desc, str))] = bug['description']
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

    # data - path
    def prepare_dataset(self, issues_by_buckets, path_train='train', path_test='test'):
        if not self.bug_set or len(self.bug_set) == 0:
            raise Exception('self.bug_set not initialized')
        # global train_data
        # global dup_sets
        # global bug_ids
        try:
            self.train_data = self.load_object('train_data')
            self.dup_sets_train = self.load_object('dup_sets_train')
            self.test_data = self.load_object('test_data')
            self.dup_sets_test = self.load_object('dup_sets_test')
            self.bug_ids = self.load_object('bug_ids')
        except:
            self.train_data, self.dup_sets_train = Baseline.read_train_data(issues_by_buckets, self.DIR, list(self.bug_set), path_train)
            self.test_data, self.dup_sets_test = Baseline.read_test_data(self.DIR, list(self.bug_set), issues_by_buckets, path_test)
            self.bug_ids = Baseline.read_bug_ids(self.DIR)
            self.save_object('train_data', self.train_data)
            self.save_object('dup_sets_train', self.dup_sets_train)
            self.save_object('test_data', self.test_data)
            self.save_object('dup_sets_test', self.dup_sets_test)
            self.save_object('bug_ids', self.bug_ids)

    def load_object(self, path):
        with open(os.path.join(self.DIR, '{}.pkl'.format(path)), 'rb') as f:
            return pickle.load(f)
    def save_object(self, path, obj):
        with open(os.path.join(self.DIR, '{}.pkl'.format(path)), 'wb') as f:
            pickle.dump(obj, f)

    def to_one_hot(self, idx, size):
        one_hot = np.zeros(size)
        one_hot[int(float(idx))] = 1
        return one_hot

    def data_padding(self, data, max_seq_length, method):
        seq_lengths = [len(seq) for seq in data]
        seq_lengths.append(6)
        #max_seq_length = min(max(seq_lengths), max_seq_length)
        padded_data = np.zeros(shape=[len(data), max_seq_length])
        for i, seq in enumerate(data):
            seq = seq[:max_seq_length]
            end_sent = -1
            for j, token in enumerate(seq):
                if(int(token) == self.TOKEN_END):
                    token = 0
                padded_data[i, j] = int(token)
            if method == 'bert':
                padded_data[i] = np.concatenate([padded_data[i][:-1], [self.TOKEN_END]])
        return padded_data.astype(np.int)

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

    def read_batch_bugs_centroid(self, batch, bug):
        batch['centroid_embed'].append(bug['centroid_embed'])

    def get_neg_bug_semihard(self, retrieval, model, batch_bugs, anchor, pos, invalid_bugs, method='keras'):
        if method == 'keras':
            vector = model.predict([ np.array([self.bug_set[anchor]['title_token']]), 
                                    np.array([self.bug_set[anchor]['description_token']]), 
                                    np.array([retrieval.get_info(self.bug_set[anchor])]) ])
        elif method == 'bert':
            vector = model.predict([ np.array([self.bug_set[anchor]['title_token']]),
                                    np.zeros_like(np.array([self.bug_set[anchor]['title_token']])), 
                                    np.array([self.bug_set[anchor]['description_token']]), 
                                    np.zeros_like(np.array([self.bug_set[anchor]['description_token']])),
                                    np.array([retrieval.get_info(self.bug_set[anchor])]) ])
        annoy = AnnoyIndex(vector.shape[1])
        embeds = []
        title_data, desc_data, info_data = [], [], []
        batch_bugs_wo_positives = list(set(batch_bugs) - set(invalid_bugs))
        #batch_bugs_wo_positives.append(pos)
        for bug_id in batch_bugs_wo_positives:
            bug = self.bug_set[bug_id]
            title_data.append(bug['title_token'])
            desc_data.append(bug['description_token'])
            info_data.append(retrieval.get_info(bug))
        if method == 'keras':
            embeds = model.predict([ np.array(title_data), np.array(desc_data), np.array(info_data) ])
        elif method == 'bert':
            embeds = model.predict([ np.array(title_data), np.zeros_like(title_data), np.array(desc_data), np.zeros_like(desc_data), np.array(info_data) ])
        for bug_id, embed in zip(batch_bugs_wo_positives, embeds):
            annoy.add_item(bug_id, embed)
        annoy.build(10) # 10 trees
        num_of_examples = 20
        rank = annoy.get_nns_by_vector(vector[0], num_of_examples, include_distances=False)
        neg_bug = rank[0]
        if neg_bug == anchor:
            neg_bug = rank[1]
        # Getting the next position from pos
        # indice = 0
        # found_pos = -1
        # neg_bug = -1
        # while indice < num_of_examples:
        #     neg_bug = rank[indice]
        #     if neg_bug == pos:
        #         found_pos = neg_bug
        #     elif found_pos != -1 and neg_bug != -1:
        #         break
        #     indice+=1
        # if(found_pos == neg_bug):
        #     neg_bug = rank[-2]
        return neg_bug

    def fill_padding(self, bug, window_padding, pad_desc):
        vector_padding = bug['description_token'][window_padding:window_padding+pad_desc]
        if(len(vector_padding) != pad_desc):
            return
        bug['description_token'] = np.concatenate([[self.TOKEN_BEGIN], vector_padding[1:-1], [self.TOKEN_END]])

    def apply_window_padding(self, bug_anchor, bug_neg):
        pad_title = self.MAX_SEQUENCE_LENGTH_T
        pad_desc = self.MAX_SEQUENCE_LENGTH_D
        iteration = 1
        while np.array_equal(bug_anchor['description_token'], bug_neg['description_token']) and pad_desc * iteration < len(bug_neg['description_token']):
            size_content = len(bug_neg['description_token']) - pad_desc * iteration
            if(size_content >= pad_desc):
                window_padding = pad_desc * iteration
                self.fill_padding(bug_neg, window_padding, pad_desc)
            elif(size_content > 0):
                window_padding = pad_desc * iteration + size_content
                self.fill_padding(bug_neg, window_padding, pad_desc)
            iteration+=1

    # data - path
    # batch_size - 128
    # n_neg - 1
    def batch_iterator(self, retrieval, model, data, dup_sets, bug_ids, batch_size, n_neg, issues_by_buckets,
    TRIPLET_HARD=False, FLOATING_PADDING=False):
        # global train_data
        # global self.dup_sets
        # global self.bug_ids
        # global self.bug_set

        random.shuffle(data)

        batch_input, batch_pos, batch_neg = {'title' : [], 'desc' : [], 'info' : []}, \
                                                {'title' : [], 'desc' : [], 'info' : []}, \
                                                    {'title' : [], 'desc' : [], 'info' : []}

        n_train = len(data)

        batch_triplets, batch_bugs_anchor, batch_bugs_pos, batch_bugs_neg, batch_bugs = [], [], [], [], []

        all_bugs = bug_ids #list(issues_by_buckets.keys())
        buckets = retrieval.buckets

        for offset in range(batch_size):
            anchor, pos = data[offset][0], data[offset][1]
            batch_bugs_anchor.append(anchor)
            batch_bugs_pos.append(pos)
            batch_bugs.append(anchor)
            batch_bugs.append(pos)
            #batch_bugs += dup_sets[anchor]
        
        for anchor, pos in zip(batch_bugs_anchor, batch_bugs_pos):
            while True:
                if not TRIPLET_HARD:
                    neg = self.get_neg_bug(anchor, buckets[issues_by_buckets[anchor]], issues_by_buckets, all_bugs)
                else:
                    neg = self.get_neg_bug_semihard(retrieval, model, batch_bugs, anchor, pos, buckets[issues_by_buckets[anchor]])
                bug_anchor = self.bug_set[anchor]
                bug_pos = self.bug_set[pos]
                if neg not in self.bug_set:
                    continue
                batch_bugs.append(neg)
                bug_neg = self.bug_set[neg]
                break
            
            self.read_batch_bugs(batch_input, bug_anchor)
            self.read_batch_bugs(batch_pos, bug_pos)
            self.read_batch_bugs(batch_neg, bug_neg)

            # check padding of desc field
            if(FLOATING_PADDING):
                self.apply_window_padding(bug_anchor, bug_pos)
                self.apply_window_padding(bug_anchor, bug_neg)
                self.apply_window_padding(bug_pos, bug_neg)

            # triplet bug and master
            batch_triplets.append([anchor, pos, neg])

        batch_input['title'] = np.array(batch_input['title'])
        batch_input['desc'] = np.array(batch_input['desc'])
        batch_input['info'] = np.array(batch_input['info'])
        batch_pos['title'] = np.array(batch_pos['title'])
        batch_pos['desc'] = np.array(batch_pos['desc'])
        batch_pos['info'] = np.array(batch_pos['info'])
        batch_neg['title'] = np.array(batch_neg['title'])
        batch_neg['desc'] = np.array(batch_neg['desc'])
        batch_neg['info'] = np.array(batch_neg['info'])

        n_half = len(batch_triplets) // 2
        if n_half > 0:
            pos = np.full((1, n_half), 1)
            neg = np.full((1, n_half), 0)
            sim = np.concatenate([pos, neg], -1)[0]
        else:
            sim = np.array([np.random.choice([1, 0])])

        input_sample, input_pos, input_neg = {}, {}, {}

        input_sample = { 'title' : batch_input['title'], 'description' : batch_input['desc'], 'info' : batch_input['info'] }
        input_pos = { 'title' : batch_pos['title'], 'description' : batch_pos['desc'], 'info': batch_pos['info'] }
        input_neg = { 'title' : batch_neg['title'], 'description' : batch_neg['desc'], 'info': batch_neg['info'] }

        return batch_triplets, input_sample, input_pos, input_neg, sim #sim

    def get_bug_set(self):
        return self.bug_set

    def load_vocabulary(self, vocab_file):
        try:
            with open(vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                print('vocabulary loaded')
                return vocab
        except IOError:
            print('can not load vocabulary')
    
    def generating_embed(self, GLOVE_DIR, EMBEDDING_DIM):
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'), 'rb')
        loop = tqdm(f)
        loop.set_description("Loading Glove")
        for line in loop:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            loop.update()
        f.close()
        loop.close()

        print('Total %s word vectors in Glove 42B 300d.' % len(embeddings_index))

        vocab = self.load_vocabulary(os.path.join(self.DIR, 'word_vocab.pkl'))
        vocab_size = len(vocab)

        # Initialize uniform the vector considering the Tanh activation
        embedding_matrix = np.random.uniform(-1.0, 1.0, (vocab_size, EMBEDDING_DIM))
        embedding_matrix[0, :] = np.zeros(EMBEDDING_DIM)

        oov_count = 0
        for word, i in vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                oov_count += 1
        print('Number of OOV words: %d' % oov_count)
        self.embedding_matrix = embedding_matrix