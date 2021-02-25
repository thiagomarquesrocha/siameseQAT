import os
from keras_bert import load_vocabulary
from multiprocessing import Pool

class Util:

    BUG_PAIR_OUTPUT = 'bug_pairs'
    BUG_IDS_OUTPUT = 'bug_ids'

    @staticmethod
    def paralelize_processing(bugs, callback, parameters):
        cpu = os.cpu_count() - 1
        pool = Pool(processes=cpu) # start N worker processes
        works = []
        n = len(bugs) // cpu
        n = 1 if n == 0 else n
        sliced = []
        pos_end = n
        end = len(bugs)
        for i in range(cpu):
            pos_end = end if pos_end>=end else pos_end
            pos_end = end if (i+1) == cpu and pos_end < end else pos_end
            sliced.append(bugs[i*n:pos_end])
            pos_end += n

        print("Slicing in {} workers".format(len(sliced)))
        for s in sliced:
            if len(s) > 0:
                config = list(parameters)
                config.insert(0, s)
                config = tuple(config)
                works.append(pool.apply_async(callback, config))
                #dump_vocabulary(s, bug_dir)

        print("Executing the works...")
        res = [w.get() for w in works]
        return res

    @staticmethod
    def save_dict(set, filename):
        with open(filename, 'w') as f:
          for i, item in enumerate(set):
            f.write('%s\t%d\n' % (item, i))

    @staticmethod
    def load_dict(filename):
        dict = {}
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.split('\t')
                dict[tokens[0]] = tokens[1]
        return dict

    @staticmethod
    def getting_pairs(array):
        res = []
        bug_ids = set()
        for row in array:
            dup_bucket, dups = row
            bug_ids.add(dup_bucket)
            dups = list(dups)
            while len(dups) > 1:
                bucket = dups[0]
                bug_ids.add(bucket)
                dups.remove(bucket)
                for d in dups:
                    bug_ids.add(d)
                    res.append([bucket, d])
        return res, bug_ids   

    @staticmethod
    def read_pairs(DIR, buckets, df):
        bug_pairs = []
        bucket_dups = []
        bug_ids = set()
        # buckets
        for key in buckets:
          if len(buckets[key]) > 1:
              bucket_dups.append([key, list(buckets[key])])

        bug_pairs, bug_ids = Util.getting_pairs(bucket_dups)

        with open(os.path.join(DIR, '{}.txt'.format(Util.BUG_PAIR_OUTPUT)), 'w') as f:
          for pair in bug_pairs:
            f.write("{} {}\n".format(pair[0], pair[1]))
        bug_ids = sorted(bug_ids)
        with open(os.path.join(DIR, '{}.txt'.format(Util.BUG_IDS_OUTPUT)), 'w') as f:
          for bug_id in bug_ids:
            f.write("%d\n" % bug_id)
        return bug_pairs, bug_ids

    @staticmethod
    def pretrained_bert(pretrained_path='uncased_L-12_H-768_A-12'):
        config_path = os.path.join(pretrained_path, 'bert_config.json')
        model_path = os.path.join(pretrained_path, 'bert_model.ckpt')
        vocab_path = os.path.join(pretrained_path, 'vocab.txt')

        token_dict = load_vocabulary(vocab_path)

        return config_path, model_path, vocab_path, token_dict