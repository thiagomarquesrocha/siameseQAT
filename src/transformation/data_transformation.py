from nlp.tokenizer import Tokenizer
from utils.util import Util
from tqdm import tqdm
from collections import defaultdict
import _pickle as pickle
import os
import json

class DataTransformation:
    
    def __init__(self, config, tokenizer):
        self.DIR = config.DIR_OUTPUT
        self.BASE = config.BASE
        self.PREPROCESSING = config.PREPROCESSING
        self.TRAIN_PATH = config.TRAIN_OUTPUT
        self.MAX_SEQUENCE_LENGTH_T = config.MAX_SEQUENCE_LENGTH_T
        self.MAX_SEQUENCE_LENGTH_D = config.MAX_SEQUENCE_LENGTH_D
        
        self.tokenizer = tokenizer
        self.bugs = {}
        self.bugs_saved = []


    def select_fields_stage(self, bug_ids, df):
        print("Normalizing text...")
        products = set()
        bug_severities = set()
        priorities = set()
        versions = set()
        components = set()
        bug_statuses = set()
        text = []
        normalized_bugs_json = []
        print("Total:", df.shape[0])
        res = Util.paralelize_processing(df, self.processing_normalized_data, (self.tokenizer, ))
        for result in res:
            if self.BASE != 'firefox':
                products = products.union(result[0])
                bug_severities = bug_severities.union(result[1])
            priorities = priorities.union(result[2])
            versions = versions.union(result[3])
            components = components.union(result[4])
            bug_statuses = bug_statuses.union(result[5])
            text += result[6]
            normalized_bugs_json += result[7]
        print("Total of normalized: ", len(normalized_bugs_json))
        
        print("Writing the normalized_bugs.json")
        with open(os.path.join(self.DIR, 'normalized_bugs.json'), 'w') as f:
            for row in tqdm(normalized_bugs_json):
                f.write(row)
        
        if self.BASE != 'firefox':
            Util.save_dict(products, os.path.join(self.DIR, 'product.dic'))
            Util.save_dict(bug_severities, os.path.join(self.DIR, 'bug_severity.dic'))
        Util.save_dict(priorities, os.path.join(self.DIR, 'priority.dic'))
        Util.save_dict(versions, os.path.join(self.DIR, 'version.dic'))
        Util.save_dict(components, os.path.join(self.DIR, 'component.dic'))
        Util.save_dict(bug_statuses, os.path.join(self.DIR, 'bug_status.dic'))
        return text

    def build_vocabulary_stage(self, train_text, MAX_NB_WORDS):
        word_freq = self.build_freq_dict_stage(train_text)
        print('word vocabulary')
        word_vocab = self.save_vocab_stage(word_freq, MAX_NB_WORDS, 'word_vocab_bert.pkl')
        return word_vocab

    def save_vocab_stage(self, freq_dict, vocab_size, filename):
        top_tokens = sorted(freq_dict.items(), key=lambda x: -x[1])[:vocab_size - 2]
        print('most common token is %s which appears %d times' % (top_tokens[0][0], top_tokens[0][1]))
        print('less common token is %s which appears %d times' % (top_tokens[-1][0], top_tokens[-1][1]))
        vocab = {}
        i = 2  # 0-index is for padding, 1-index is for UNKNOWN
        for j in range(len(top_tokens)):
            vocab[top_tokens[j][0]] = i
            i += 1
        with open(os.path.join(self.DIR, filename), 'wb') as f:
            pickle.dump(vocab, f)
        return vocab
    
    def build_freq_dict_stage(self, train_text):
        print('building frequency dictionaries')
        word_freq = defaultdict(int)
        for text in tqdm(train_text):
            for word in text.split():
                word_freq[word] += 1
        return word_freq

    def load_bugs_stage(self, word_vocab, total):
        bug_dir = os.path.join(self.DIR, 'bugs')
        if not os.path.exists(bug_dir):
            os.mkdir(bug_dir)
        bugs = []
        print("Reading the normalized_bugs.json ...")
        if self.BASE != 'firefox':
            product_dict = Util.load_dict(os.path.join(self.DIR,'product.dic'))
            bug_severity_dict = Util.load_dict(os.path.join(self.DIR,'bug_severity.dic'))
        priority_dict = Util.load_dict(os.path.join(self.DIR,'priority.dic'))
        version_dict = Util.load_dict(os.path.join(self.DIR,'version.dic'))
        component_dict = Util.load_dict(os.path.join(self.DIR,'component.dic'))
        bug_status_dict = Util.load_dict(os.path.join(self.DIR,'bug_status.dic'))

        with open(os.path.join(self.DIR, 'normalized_bugs.json'), 'r') as f:
            #loop = tqdm(f)
            with tqdm(total=total) as loop:
                for line in f:
                    bug = json.loads(line)
                    if self.BASE != 'firefox':
                        bug['product'] = product_dict.get(bug['product'])
                        bug['bug_severity'] = bug_severity_dict.get(bug['bug_severity'])
                    bug['priority'] = priority_dict.get(bug['priority'])
                    bug['version'] = version_dict.get(bug['version'])
                    bug['component'] = component_dict.get(bug['component'])
                    bug['bug_status'] = bug_status_dict.get(bug['bug_status'])
                    bugs.append(bug)
                    loop.update(1)

        return bugs

    def dump_vocabulary(self, bugs, word_vocab, bug_dir):
        UNK = 1
        cont=0
        total = len(bugs)
        print("Starting the dump ...")
        bugs_set = {}
        bugs_saved = []
        for bug in tqdm(bugs):
            #bug = json.loads(line)
            #print(bug)
            cont+=1
            if self.PREPROCESSING == 'bert':
                ids, segments = self.tokenizer.encode('' if bug['description_original'] == None else bug['description_original'], max_len=self.MAX_SEQUENCE_LENGTH_D)
                bug['description_token'] = ids
                bug['description_segment'] = segments
                ids, segments = self.tokenizer.encode('' if bug['title_original'] == None else bug['title_original'], max_len=self.MAX_SEQUENCE_LENGTH_T)
                bug['title_token'] = ids
                bug['title_segment'] = segments
                bug.pop('description_original')
                bug.pop('title_original')
            else: # BASELINE
                bug['description_token'] = [word_vocab.get(w.encode('utf-8'), UNK) for w in bug['description'].split()]
                if len(bug['title']) == 0:
                    bug['title'] = bug['description'][:10]
                bug['title_token'] = [word_vocab.get(w.encode('utf-8'), UNK) for w in bug['title'].split()]
            # Save the bug processed
            bugs_set[bug['issue_id']] = bug
            with open(os.path.join(bug_dir, str(bug['issue_id']) + '.pkl'), 'wb') as f:
                pickle.dump(bug, f)
            bugs_saved.append(bug['issue_id'])

        return [bugs_set, bugs_saved]

    def processing_dump_stage(self, bugs, word_vocab, bugs_id, bugs_id_dataset):
        #clear_output()
        bug_dir = os.path.join(self.DIR, 'bugs')
        res = Util.paralelize_processing(bugs, self.dump_vocabulary, (word_vocab, bug_dir, ))
        for result in res:
            bugs_set = result[0]
            bugs_saved = result[1]
            for bug in bugs_set:
                self.bugs[bug] = bugs_set[bug]
            self.bugs_saved += bugs_saved

        self.validing_bugs_id(bugs_id, bugs_id_dataset)

    def validing_bugs_id(self, bugs_id, bugs_id_dataset):
        print("Check if all bugs id regirested in the pairs exist in dataset")
        bugs_invalid = set(bugs_id) - set(bugs_id_dataset)
        bugs_id_dataset = set(bugs_id_dataset) - bugs_invalid
        bugs_id_dataset = sorted(bugs_id_dataset)
        with open(os.path.join(self.DIR, 'bug_ids.txt'), 'w') as f:
            for bug_id in bugs_id_dataset:
                f.write("%d\n" % bug_id)
        print("Bugs not present in dataset: ", list(bugs_invalid))
        bug_pairs = []
        with open(os.path.join(self.DIR, '{}.txt'.format(self.TRAIN_PATH)), 'r') as f:
            for line in f:
                bug1, bug2 = line.strip().split()
                if bug1 not in bugs_invalid and bug2 not in bugs_invalid:
                    bug_pairs.append([bug1, bug2])
        with open(os.path.join(self.DIR, '{}.txt'.format(self.TRAIN_PATH)), 'w') as f:
            for pairs in bug_pairs:
                f.write("{} {}\n".format(pairs[0], pairs[1]))

    def save_bugs_stage(self, word_vocab, bug_ids, bugs_id_dataset):
        num_lines =  len(open(os.path.join(self.DIR, 'normalized_bugs.json'), 'r').read().splitlines()) * 2
        total = num_lines // 2
        bugs = self.load_bugs_stage(word_vocab, total)
        self.processing_dump_stage(bugs, word_vocab, bug_ids, bugs_id_dataset)

    def processing_normalized_data(self, df, tokenizer: Tokenizer):
        products = set()
        bug_severities = set()
        priorities = set()
        versions = set()
        components = set()
        bug_statuses = set()
        text = []
        normalized_bugs_json = []
        with tqdm(total=df.shape[0]) as loop:
            for row in df.iterrows():
                bug = row[1]
                if self.BASE != 'firefox':
                    products.add(bug['product'])
                    bug_severities.add(bug['bug_severity'])
                priorities.add(bug['priority'])
                versions.add(bug['version'])
                components.add(bug['component'])
                bug_statuses.add(bug['bug_status'])
                
                if 'description' not in bug or bug['description'] == '':
                    bug['description'] = bug['title']

                if 'title' not in bug or bug['title'] == '':
                    bug['title'] = bug['description']
                
                if self.PREPROCESSING == 'bert':
                    description = tokenizer.apply(bug['description'])
                    bug['description_original'] = bug['description']
                    bug['description'] = description
                    title = tokenizer.apply(bug['title'])
                    bug['title_original'] = bug['title']
                    bug['title'] = title
                else:
                    bug['description'] = tokenizer.apply(bug['description'])
                    bug['title'] = tokenizer.apply(bug['title'])
                    
                normalized_bugs_json.append('{}\n'.format(bug.to_json()))

                text.append(bug['description'])
                text.append(bug['title'])
                loop.update(1)

        return [products, bug_severities, priorities, versions, components, bug_statuses, text, normalized_bugs_json]

    