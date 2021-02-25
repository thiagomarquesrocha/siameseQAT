import random
import os

class Splitter:

     @staticmethod
     def split_train_test(DIR, TRAIN_PATH, TEST_PATH, bug_pairs, VALIDATION_SPLIT):
        random.shuffle(bug_pairs)
        split_idx = int(len(bug_pairs) * VALIDATION_SPLIT)
        with open(os.path.join(DIR, '{}.txt'.format(TRAIN_PATH)), 'w') as f:
            for pair in bug_pairs[:split_idx]:
                f.write("{} {}\n".format(pair[0], pair[1]))
        test_data = {}
        for pair in bug_pairs[split_idx:]:
            bug1 = int(pair[0])
            bug2 = int(pair[1])
            if bug1 not in test_data:
                test_data[bug1] = set()
            test_data[bug1].add(bug2)
        with open(os.path.join(DIR, '{}.txt'.format(TEST_PATH)), 'w') as f:
            for bug in test_data.keys():
                f.write("{} {}\n".format(bug, ' '.join([str(x) for x in test_data[bug]])))
        print('Train and test created')