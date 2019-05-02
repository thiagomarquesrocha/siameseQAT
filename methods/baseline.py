from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import sklearn.metrics
import _pickle as pickle
from sklearn.manifold import TSNE
import random
import numpy as np
from tqdm import tqdm

from keras.utils import to_categorical
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

class Baseline:

    def __init__(self, DIR, MAX_SEQUENCE_LENGTH_T, MAX_SEQUENCE_LENGTH_D):
        self.sentence_dict = {}
        self.corpus = []
        self.bug_ids = []

        self.train_data = None
        self.dup_sets = None
        self.bug_set = {}

        self.DIR = DIR
        self.GLOVE_DIR = ""
        self.MAX_SEQUENCE_LENGTH_T = MAX_SEQUENCE_LENGTH_T
        self.MAX_SEQUENCE_LENGTH_D = MAX_SEQUENCE_LENGTH_D
        self.info_dict = {'bug_severity': 7, 'bug_status': 3, 'component': 323, 'priority': 5, 'product': 116, 'version': 197}

    def load_ids(self, DIR):
        self.bug_ids = []
        with open(os.path.join(DIR, 'bug_ids.txt'), 'r') as f:
            for row in f:
                self.bug_ids.append(int(row))

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
    def save_model(DIR, model, name):
        m_dir = os.path.join(DIR, 'modelos')
        if not os.path.exists(m_dir):
            os.mkdir(m_dir)
        # serialize model to JSON
        model_json = model.to_json()
        with open(os.path.join(m_dir, "model_{}.json".format(name)), 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join(m_dir, "model_{}.h5".format(name)))
        print("Saved model to disk")

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
                        valid_input_sample['description'], valid_input_pos['description'], valid_input_neg['description']], verbose = False, 
                                                batch_size=batch_size)

        model_pos  = similarity_model.get_layer('merge_features_pos').output
        model_final = Model(inputs=similarity_model.input, outputs=model_pos)
        x_test_features_pos = model_final.predict([valid_input_sample['title'], valid_input_pos['title'], valid_input_neg['title'], 
                        valid_input_sample['description'], valid_input_pos['description'], valid_input_neg['description']], verbose = False, 
                                                batch_size=batch_size)

        model_neg  = similarity_model.get_layer('merge_features_neg').output
        model_final = Model(inputs=similarity_model.input, outputs=model_neg)
        x_test_features_neg = model_final.predict([valid_input_sample['title'], valid_input_pos['title'], valid_input_neg['title'], 
                        valid_input_sample['description'], valid_input_pos['description'], valid_input_neg['description']], verbose = False, 
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

    @staticmethod
    def padding_embed(max_char, field, bug):
        n = len(bug[field])
        if (max_char - n) > 0: # desc or title
            embed = np.empty(max_char - n)
            embed.fill(0)
            embed = np.concatenate([embed, bug[field]], axis=-1)
            embed = embed.astype(int)
        else:
            embed = np.array(bug[field][:max_char])
        return embed

    def load_preprocess(self):   
        for bug_id in tqdm(self.bug_ids):
            bug = pickle.load(open(os.path.join(self.DIR, 'bugs', '{}.pkl'.format(bug_id)), 'rb'))
        #     print(str(bug['title_word']))
            title = Baseline.padding_embed(self.MAX_SEQUENCE_LENGTH_T, 'title_word', bug)
            desc = Baseline.padding_embed(self.MAX_SEQUENCE_LENGTH_D, 'description_word', bug)
            #print(len(title), len(desc))
            #print(",".join(title.astype(str)))
            self.sentence_dict[",".join(title.astype(str))] = bug['title']
            self.sentence_dict[",".join(desc.astype(str))] = bug['description']
            self.corpus.append(bug['title'])
            self.corpus.append(bug['description'])
        #     break

    @staticmethod
    def get_neg_bug(invalid_bugs, bug_ids):
        neg_bug = random.choice(bug_ids)
        while neg_bug in invalid_bugs:
            neg_bug = random.choice(bug_ids)
        return neg_bug

    @staticmethod
    def read_test_data(data):
        test_data = []
        bug_ids = set()
        with open(os.path.join(data, 'test.txt'), 'r') as f:
            for line in f:
                tokens = line.strip().split()
                test_data.append([int(tokens[0]), [int(bug) for bug in tokens[1:]]])
                for token in tokens:
                    bug_ids.add(int(token))
        return test_data, list(bug_ids)

    @staticmethod
    def read_train_data(data):
        data_pairs = []
        data_dup_sets = {}
        print('Reading train data')
        with open(os.path.join(data, 'train.txt'), 'r') as f:
            for line in f:
                bug1, bug2 = line.strip().split()
                data_pairs.append([int(bug1), int(bug2)])
                if int(bug1) not in data_dup_sets.keys():
                    data_dup_sets[int(bug1)] = set()
                data_dup_sets[int(bug1)].add(int(bug2))
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
    def prepare_dataset(self, data):
        # global train_data
        # global dup_sets
        # global bug_ids
        if not self.train_data:
            self.train_data, self.dup_sets = Baseline.read_train_data(data)
            #print(len(train_data))
        if not self.bug_ids:
            self.bug_ids = Baseline.read_bug_ids(data)

    def siam_gen(self, data, batch_size, n_neg):
        while True:
            input_sample, input_pos, input_neg, sim = self.batch_iterator(data, batch_size, n_neg)
            yield ({ 'title_in' : input_sample['title'], 'title_pos': input_pos['title'], 'title_neg' : input_neg['title'],
            'desc_in' : input_sample['description'], 'desc_pos' : input_pos['description'], 'desc_neg' : input_neg['description'] }, sim)

    def to_one_hot(self, idx, size):
        one_hot = np.zeros(size)
        one_hot[int(float(idx))] = 1
        return one_hot

    @staticmethod
    def data_padding_bug(seq, max_seq_length):
        seq = seq[:max_seq_length]
        padding = max_seq_length - len(seq)
        if padding > 0:
            embed = np.empty(padding)
            embed.fill(0)
            return np.concatenate([embed, seq], -1).astype(int)
        else:
            return np.array(seq).astype(int)

    @staticmethod
    def data_padding(data, max_seq_length):
        padded_data = np.zeros(shape=[len(data), max_seq_length])
        for i, seq in enumerate(data):
            seq = seq[:max_seq_length]
            padding_end = max_seq_length - len(seq)
            #print(seq)
            embed = np.empty(padding_end)
            embed.fill(0)
            padded_data[i] = np.concatenate([embed, seq], -1)
        return padded_data.astype(np.int)

    def read_batch_bugs(self, batch, bug):
        info_ = np.concatenate((
            self.to_one_hot(bug['bug_severity'], self.info_dict['bug_severity']),
            self.to_one_hot(bug['bug_status'], self.info_dict['bug_status']),
            self.to_one_hot(bug['component'], self.info_dict['component']),
            self.to_one_hot(bug['priority'], self.info_dict['priority']),
            self.to_one_hot(bug['product'], self.info_dict['product']),
            self.to_one_hot(bug['version'], self.info_dict['version']))
        )
        info.append(info_)
        batch['info'].append(info)
        batch['title'].append(bug['title_word'])
        batch['desc'].append(bug['description_word'])

    # data - path
    # batch_size - 128
    # n_neg - 1
    def batch_iterator(self, data, batch_size, n_neg):
        # global train_data
        # global self.dup_sets
        # global self.bug_ids
        # global self.bug_set

        random.shuffle(self.train_data)

        batch_input, batch_pos, batch_neg = {'title' : [], 'desc' : []}, {'title' : [], 'desc' : []}, {'title' : [], 'desc' : []}

        n_train = len(self.train_data)

        batch_triplets = []
        
        for offset in range(batch_size):
            neg_bug = Baseline.get_neg_bug(self.dup_sets[self.train_data[offset][0]], self.bug_ids)
            anchor, pos, neg = self.train_data[offset][0], self.train_data[offset][1], neg_bug
            bug_anchor = self.bug_set[anchor]
            bug_pos = self.bug_set[pos]
            bug_neg = self.bug_set[neg]
            self.read_batch_bugs(batch_input, bug_anchor)
            self.read_batch_bugs(batch_pos, bug_pos)
            self.read_batch_bugs(batch_neg, bug_neg)
            batch_triplets.append([self.train_data[offset][0], self.train_data[offset][1], neg_bug])

        batch_input['title'] = np.array(batch_input['title'])
        batch_input['desc'] = np.array(batch_input['desc'])
        batch_input['info'] = np.array(batch_input['info'])
        batch_pos['title'] = np.array(batch_pos['title'])
        batch_pos['desc'] = np.array(batch_pos['desc'])
        batch_pos['info'] = np.array(batch_pos['info'])
        batch_neg['title'] = np.array(batch_neg['title'])
        batch_neg['desc'] = np.array(batch_neg['desc'])
        batch_neg['info'] = np.array(batch_neg['info'])

        n_half = batch_size // 2
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

        return input_sample, input_pos, input_neg, sim #sim

    def load_bugs(self):
        self.bug_set = {}

        loop = tqdm(total=len(self.bug_ids))

        for bug_id in self.bug_ids:
            self.bug_set[bug_id] = pickle.load(open(os.path.join(self.DIR, 'bugs', '{}.pkl'.format(bug_id)), 'rb'))
            self.bug_set[bug_id]['description_word'] = Baseline.data_padding_bug(self.bug_set[bug_id]['description_word'], self.MAX_SEQUENCE_LENGTH_D)
            self.bug_set[bug_id]['title_word'] = Baseline.data_padding_bug(self.bug_set[bug_id]['title_word'], self.MAX_SEQUENCE_LENGTH_T)
            loop.update(1)
        loop.close()

    def get_bug_set(self):
        return self.bug_set

    def display_batch(self, groups, nb):
        input_sample, input_pos, input_neg, v_sim = self.batch_iterator(groups, nb, 1)

        t_a, t_b, d_a, d_b = [], [], [], []
        
        t_a = input_sample['title']
        t_b = input_pos['title']
        d_a = input_sample['description']
        d_b = input_pos['description']
        
        for t_a, t_b_pos, t_b_neg, d_a, d_b_pos, d_b_neg, sim in zip(input_sample['title'], input_pos['title'], input_neg['title'], 
                                                                input_sample['description'], input_pos['description'], input_neg['description'], v_sim):
            

            t_b = t_b_pos if sim == 1 else t_b_neg
            d_b = d_b_pos if sim == 1 else t_b_neg
            
            #print(t_a.shape)
            key_t_a = ','.join(t_a.astype(str))
            key_t_b = ','.join(t_b.astype(str))
            key_d_a = ','.join(d_a.astype(str))
            key_d_b = ','.join(d_b.astype(str))
            print("Title: \n{}".format(self.sentence_dict[key_t_a]))
            print("Title: \n{}".format(self.sentence_dict[key_t_b]))
            print("Description: \n{}".format(self.sentence_dict[key_d_a]))
            print("Description: \n{}".format(self.sentence_dict[key_d_b]))
            print("similar =", str(sim))
            print("########################")

    def word_index_count(self, corpus, MAX_NB_WORDS):
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(corpus)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        
        return word_index

    def generating_embed(self, GLOVE_DIR, EMBEDDING_DIM, MAX_NB_WORDS):
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'), 'rb')
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors in Glove 42B 300d.' % len(embeddings_index))

        self.word_index = self.word_index_count(self.corpus, MAX_NB_WORDS)

        self.embedding_matrix = np.random.random((len(self.word_index) + 1, EMBEDDING_DIM))
        for word, i in tqdm(self.word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    ############################# CUSTOM LOSS #####################################
    @staticmethod
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())

    # https://github.com/keras-team/keras/issues/3031
    # https://github.com/keras-team/keras/issues/8335
    @staticmethod
    def cosine_distance(inputs):
        x, y = inputs
        x = Baseline.l2_normalize(x, axis=-1)
        y = Baseline.l2_normalize(y, axis=-1)
        similarity = K.batch_dot(x, y, axes=1)
        distance = K.constant(1) - similarity
        # Distance goes from 0 to 2 in theory, but from 0 to 1 if x and y are both
        # positive (which is the case after ReLU activation).
        return K.mean(distance, axis=-1)
    @staticmethod
    def margin_loss(y_true, y_pred):
        margin = K.constant(1.0)
        return K.mean(K.maximum(0.0, margin - y_pred[0] + y_pred[1]))
    @staticmethod
    def pos_distance(y_true, y_pred):
        return K.mean(y_pred[0])
    @staticmethod
    def neg_distance(y_true, y_pred):
        return K.mean(y_pred[1])
    @staticmethod
    def stack_tensors(vects):
        return K.squeeze(K.stack(vects),axis=1) # stack adds a new dim. So squeeze it
        # better method is to use concatenate
        return K.concatenate(vects,axis=1)