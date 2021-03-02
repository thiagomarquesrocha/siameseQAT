import numpy as np
import logging
from utils.util import Util

logger = logging.getLogger('Recall')

class Recall:

    PRECISION = 3

    def __init__(self):
        pass

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.MAX_RANK = 25
    
    def verbose(self, verbose=1):
        self.verbose = verbose

    """
        Rank recall_rate_@k
        rank = "query:master|master:id:sim,master:id:sim"
    """
    def top_k_recall(self, row, k):
        query, rank = row.split('|')
        query_dup_id, ground_truth = query.split(":")
        ground_truth = np.asarray(ground_truth.split(','), int)
        candidates = [int(item.split(':')[0]) for pos, item in enumerate(rank.split(",")[:self.MAX_RANK])]
        corrects = len(set(ground_truth) & set(candidates[:k]))
        # relevant_queries = k if(len(ground_truth) > k) else len(ground_truth)
        # corrects = corrects / relevant_queries
        corrects = 1 if corrects > 0 else 0
        total = 1
        return float(corrects), total

    def evaluate(self, path):
        
        self.recall_at_1,\
        self.recall_at_5, self.recall_at_10, \
        self.recall_at_15, self.recall_at_20, self.recall_at_25 = [], [], [], [], [], []

        self.recall_at_1_corrects_sum,\
        self.recall_at_5_corrects_sum, self.recall_at_10_corrects_sum, \
        self.recall_at_15_corrects_sum, self.recall_at_20_corrects_sum, self.recall_at_25_corrects_sum = 0, 0, 0, 0, 0, 0
        
        self.recall_at_1_total_sum, \
        self.recall_at_5_total_sum, self.recall_at_10_total_sum, self.recall_at_15_total_sum, \
        self.recall_at_20_total_sum, self.recall_at_25_total_sum = 0, 0, 0, 0, 0, 0
        if(self.verbose):
            logger.debug("Evaluating...")
        if type(path) == str:
            with open(path, 'r') as file_input:
                for row in file_input:
                    self.recall(row)
        else:
            for row in path:
                self.recall(row)
        
        report = {
            '0 - recall_at_1' : round(self.recall_at_1_corrects_sum / self.recall_at_1_total_sum, self.PRECISION),
            '1 - recall_at_5' : round(self.recall_at_5_corrects_sum / self.recall_at_5_total_sum, self.PRECISION),
            '2 - recall_at_10' : round(self.recall_at_10_corrects_sum / self.recall_at_10_total_sum, self.PRECISION),
            '3 - recall_at_15' : round(self.recall_at_15_corrects_sum / self.recall_at_15_total_sum, self.PRECISION),
            '4 - recall_at_20' : round(self.recall_at_20_corrects_sum / self.recall_at_20_total_sum, self.PRECISION),
            '5 - recall_at_25' : round(self.recall_at_25_corrects_sum / self.recall_at_25_total_sum, self.PRECISION)
        }

        if(self.verbose):
            self.display_rank_evaluation(report)

        return report

    def display_rank_evaluation(self, report):
        for key, value in Util.sort_dict_by_key(report).items():
            logger.debug("{} = {}".format(key, value))

    def get_recalls(self):
        report = {
            '0 - recall_at_1' : self.recall_at_1,
            '1 - recall_at_5' : self.recall_at_5,
            '2 - recall_at_10' : self.recall_at_10,
            '3 - recall_at_15' : self.recall_at_15,
            '4 - recall_at_20' : self.recall_at_20,
            '5 - recall_at_25' : self.recall_at_25
        }

        return report
    
    def recall(self, row):
        #if row == '': continue
        self.recall_at_1_corrects, self.recall_at_1_total = self.top_k_recall(row, k=1)
        self.recall_at_5_corrects, self.recall_at_5_total = self.top_k_recall(row, k=5)
        self.recall_at_10_corrects, self.recall_at_10_total = self.top_k_recall(row, k=10)
        self.recall_at_15_corrects, self.recall_at_15_total = self.top_k_recall(row, k=15)
        self.recall_at_20_corrects, self.recall_at_20_total = self.top_k_recall(row, k=20)
        self.recall_at_25_corrects, self.recall_at_25_total = self.top_k_recall(row, k=25)

        self.recall_at_1 += [self.recall_at_1_corrects]
        self.recall_at_5 += [self.recall_at_5_corrects]
        self.recall_at_10 += [self.recall_at_10_corrects]
        self.recall_at_15 += [self.recall_at_15_corrects]
        self.recall_at_20 += [self.recall_at_20_corrects]
        self.recall_at_25 += [self.recall_at_25_corrects]

        self.recall_at_1_corrects_sum += self.recall_at_1_corrects
        self.recall_at_5_corrects_sum += self.recall_at_5_corrects
        self.recall_at_10_corrects_sum += self.recall_at_10_corrects
        self.recall_at_15_corrects_sum += self.recall_at_15_corrects
        self.recall_at_20_corrects_sum += self.recall_at_20_corrects
        self.recall_at_25_corrects_sum += self.recall_at_25_corrects

        self.recall_at_1_total_sum += self.recall_at_1_total
        self.recall_at_5_total_sum += self.recall_at_5_total
        self.recall_at_10_total_sum += self.recall_at_10_total
        self.recall_at_15_total_sum += self.recall_at_15_total
        self.recall_at_20_total_sum += self.recall_at_20_total
        self.recall_at_25_total_sum += self.recall_at_25_total