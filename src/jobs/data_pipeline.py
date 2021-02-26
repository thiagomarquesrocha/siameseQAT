import os
import pandas as pd
import networkx as nx
import _pickle as pickle
import logging

from tqdm import tqdm
from utils.util import Util
from utils.splitter import Splitter
from nlp.tokenizer_base import TokenizerBase
from nlp.tokenizer_bert import TokenizerBert
from transformation.data_transformation import DataTransformation

logger = logging.getLogger('DataPipeline')

class DataPipeline:
    def __init__(self, DATASET, DOMAIN, COLAB, 
                    PREPROCESSING, VALIDATION_SPLIT=0.9,
                    MAX_SEQUENCE_LENGTH_TITLE=50, MAX_SEQUENCE_LENGTH_DESC=150):
        self.MAX_NB_WORDS = 20000
        self.MAX_SEQUENCE_LENGTH_T = MAX_SEQUENCE_LENGTH_TITLE
        self.MAX_SEQUENCE_LENGTH_D = MAX_SEQUENCE_LENGTH_DESC
        
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.COLAB = COLAB
        self.PREPROCESSING = PREPROCESSING
        self.DATASET=DATASET
        self.DOMAIN=DOMAIN

        self.TRAIN_OUTPUT = 'train_chronological'
        self.TEST_OUTPUT = 'test_chronological'
        self.DIR_OUTPUT = os.path.join('{}data'.format(COLAB), 'processed') # where will be exported
        self.NORMALIZED_DIR = os.path.join('{}data'.format(self.COLAB), 'normalized', self.DATASET)

    def create_bucket(self, df):
        print("Creating the buckets...")
        buckets = {}
        G=nx.Graph()
        for row in tqdm(df.iterrows()):
            bug_id = row[1]['issue_id']
            dup_id = row[1]['dup_id']
            if dup_id == '[]':
                G.add_node(bug_id)
            else:
                G.add_edges_from([(int(bug_id), int(dup_id))])
        for g in tqdm(nx.connected_components(G)):
            group = set(g)
            for bug in g:
                master = int(bug)
                query = df[df['issue_id'] == master]
                if query.shape[0] <= 0:
                    group.remove(master)
                    master = np.random.choice(list(group), 1)
            buckets[int(master)] = group
        return buckets

    def save_buckets(self, buckets):
        with open(os.path.join(self.DIR_OUTPUT, self.BASE + '_buckets.pkl'), 'wb') as f:
            pickle.dump(buckets, f)

    def setup(self):
        # create 'dataset' directory
        bug_dir = os.path.join(self.DIR_OUTPUT, self.DATASET)
        if not os.path.exists(self.DIR_OUTPUT):
            os.mkdir(self.DIR_OUTPUT)
        if not os.path.exists(bug_dir):
            os.mkdir(bug_dir)

        # create 'processing' directory
        bug_dir = os.path.join(bug_dir, self.PREPROCESSING)
        if not os.path.exists(bug_dir):
            os.mkdir(bug_dir)
        
        logger.debug("DIR_OUTPUT = {}".format(bug_dir))

        self.BASE = self.DATASET
        self.DIR_OUTPUT = bug_dir
        self.DOMAIN = os.path.join(self.NORMALIZED_DIR, self.DOMAIN)

    def load_train(self):
        logger.debug('Loading : {}'.format(self.DOMAIN))
        df_train = pd.read_csv('{}.csv'.format(self.DOMAIN))
        if self.BASE != 'firefox':
            df_train.columns = ['issue_id','bug_severity','bug_status','component',
                                'creation_ts','delta_ts','description','dup_id','priority',
                                'product','resolution','title','version']
        else:
            df_train.columns = ['issue_id','priority','component','dup_id','title',
                                    'description','bug_status','resolution','version',
                                        'creation_ts', 'delta_ts']
        return df_train

    def run(self):
        
        # Setup directories
        self.setup()
        # Train
        df_train = self.load_train()
        # Create buckets
        buckets = self.create_bucket(df_train)
        self.save_buckets(buckets)
        # Create pairs
        bug_pairs, bug_ids = Util.read_pairs(self.DIR_OUTPUT, buckets, df_train)
        bugs_id_dataset = df_train['issue_id'].values
        logger.debug("Number of duplicate: {}".format(len(bug_ids)))
        logger.debug("Number of pairs: {}".format(len(bug_pairs)))

        # Split into train/test
        Splitter.split_train_test(self.DIR_OUTPUT, self.TRAIN_OUTPUT, 
                                    self.TEST_OUTPUT, bug_pairs, 
                                        self.VALIDATION_SPLIT)

        # Transformation config
        tokenizer = TokenizerBert() if self.PREPROCESSING == 'bert' else TokenizerBase()
        # Create pipeline
        data_pipeline = DataTransformation(self, tokenizer)
        # Step to select what fields to process
        text = data_pipeline.select_fields_stage(bug_ids, df_train)
        # Step to build the vocab
        word_vocab = data_pipeline.build_vocabulary_stage(text, self.MAX_NB_WORDS)
        # Step to preprocess the dump of bugs
        data_pipeline.save_bugs_stage(word_vocab, bug_ids, bugs_id_dataset)
