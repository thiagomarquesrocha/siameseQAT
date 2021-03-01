import random
import os
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('Splitter')

class Splitter:
     
    @staticmethod
    def count_bucket_size(bucket_stats, buckets, split, split_bucket):
        for bucket_id in split_bucket:
            bucket_size = len(buckets[bucket_id])
            if bucket_size in bucket_stats[split]['bucket_size']:
                bucket_stats[split]['bucket_size'][bucket_size] += 1
            else:
                bucket_stats[split]['bucket_size'][bucket_size] = 1

    @staticmethod
    def create_pairs(buckets, name, set_buckets, pair_dups, ids):
        for bucket_id in set_buckets:
            bucket = list(buckets[bucket_id])
            for i, bug_id in enumerate(bucket):
                for dup_id in bucket[i+1:]:
                    pair_dups.append([bug_id, dup_id])
                    ids.append(bug_id)
                    ids.append(dup_id)

    @staticmethod
    def get_buckets_for_bugs(buckets):
        issues_by_buckets = {}
        for bucket in tqdm(buckets):
            issues_by_buckets[bucket] = bucket
            for issue in np.array(buckets[bucket]).tolist():
                issues_by_buckets[issue] = bucket
        return issues_by_buckets

    @staticmethod
    def split_train_test(DIR, TRAIN_PATH, TEST_PATH, buckets, VALIDATION_SPLIT):
        """
            Split train and test based on clusters
        """
        bucket_stats = { 'train' : { 'bucket_size' : {} }, 'test' : { 'bucket_size' : {} } }
        list_of_buckets = list(buckets.keys())
        random.shuffle(list_of_buckets)
        SPLIT_SIZE = int(len(list_of_buckets) * VALIDATION_SPLIT)
        train_buckets = list_of_buckets[:SPLIT_SIZE]
        test_buckets = list_of_buckets[SPLIT_SIZE:]

        logger.debug("Duplicate groups in train: {}".format(len(train_buckets)))
        logger.debug("Duplicate groups in test: {}".format(len(test_buckets)))

        logger.debug('Train and test created')

        Splitter.count_bucket_size(bucket_stats, buckets, 'train', train_buckets)
        Splitter.count_bucket_size(bucket_stats, buckets, 'test', test_buckets)

        train_dups = []
        test_dups = []
        train_ids = []
        test_ids = []

        Splitter.create_pairs(buckets, "train", train_buckets, train_dups, train_ids)
        Splitter.create_pairs(buckets, "test", test_buckets, test_dups, test_ids)
        logger.debug("Train pair size {}".format(len(train_dups)))
        logger.debug("Test pair size {}".format(len(test_dups)))
        logger.debug("******* IDS ***********")
        logger.debug("Train ids size {}".format(len(train_ids)))
        logger.debug("Test ids size {}".format(len(test_ids)))

        # Train
        with open(os.path.join(DIR, "{}.txt".format(TRAIN_PATH)), 'w') as f:
            for pair in train_dups:
                f.write("{} {}\n".format(pair[0], pair[1]))
        # Test
        test_data = {}
        issues_by_buckets = Splitter.get_buckets_for_bugs(buckets)
        for pair in test_dups:
            bug1 = int(pair[0])
            bug2 = int(pair[1])
            
            bucket = issues_by_buckets[bug1]
                
            if bucket not in test_data:
                test_data[bucket] = set()
                
            test_data[bucket].add(bug1)
            test_data[bucket].add(bug2)
        with open(os.path.join(DIR, "{}.txt".format(TEST_PATH)), 'w') as f:
            for bug in test_data.keys():
                f.write("{}\n".format(' '.join([str(x) for x in test_data[bug]])))