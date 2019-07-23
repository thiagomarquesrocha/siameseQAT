import numpy as np
import pprint

class Evaluation():
    def __init__(self, verbose=1):
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
        candidates = [int(item.split(':')[0]) for pos, item in enumerate(rank.split(",")[:self.MAX_RANK])]
        corrects = len(set([int(ground_truth)]) & set(candidates[:k]))
        total = len([ground_truth]) # only one master from query
        return float(corrects), total

    def evaluate(self, path):
        self.recall_at_5_corrects_sum, self.recall_at_10_corrects_sum, \
        self.recall_at_15_corrects_sum, self.recall_at_20_corrects_sum, self.recall_at_25_corrects_sum = 0, 0, 0, 0, 0
        self.recall_at_5_total_sum, self.recall_at_10_total_sum, self.recall_at_15_total_sum, \
        self.recall_at_20_total_sum, self.recall_at_25_total_sum = 0, 0, 0, 0, 0 
        if(self.verbose):
            print("Evaluating...")
        if type(path) == str:
            with open(path, 'r') as file_input:
                for row in file_input:
                    self.recall(row)
        else:
            for row in path:
                self.recall(row)
        
        report = {
            '1 - recall_at_5' : round(self.recall_at_5_corrects_sum / self.recall_at_5_total_sum, 2),
            '2 - recall_at_10' : round(self.recall_at_10_corrects_sum / self.recall_at_10_total_sum, 2),
            '3 - recall_at_15' : round(self.recall_at_15_corrects_sum / self.recall_at_15_total_sum, 2),
            '4 - recall_at_20' : round(self.recall_at_20_corrects_sum / self.recall_at_20_total_sum, 2),
            '5 - recall_at_25' : round(self.recall_at_25_corrects_sum / self.recall_at_25_total_sum, 2)
        }

        return report
    def recall(self, row):
        #if row == '': continue
        self.recall_at_5_corrects, self.recall_at_5_total = self.top_k_recall(row, k=5)
        self.recall_at_10_corrects, self.recall_at_10_total = self.top_k_recall(row, k=10)
        self.recall_at_15_corrects, self.recall_at_15_total = self.top_k_recall(row, k=15)
        self.recall_at_20_corrects, self.recall_at_20_total = self.top_k_recall(row, k=20)
        self.recall_at_25_corrects, self.recall_at_25_total = self.top_k_recall(row, k=25)

        self.recall_at_5_corrects_sum += self.recall_at_5_corrects
        self.recall_at_10_corrects_sum += self.recall_at_10_corrects
        self.recall_at_15_corrects_sum += self.recall_at_15_corrects
        self.recall_at_20_corrects_sum += self.recall_at_20_corrects
        self.recall_at_25_corrects_sum += self.recall_at_25_corrects

        self.recall_at_5_total_sum += self.recall_at_5_total
        self.recall_at_10_total_sum += self.recall_at_10_total
        self.recall_at_15_total_sum += self.recall_at_15_total
        self.recall_at_20_total_sum += self.recall_at_20_total
        self.recall_at_25_total_sum += self.recall_at_25_total

if __name__ == '__main__':
    evaluation = Evaluation()
    report = evaluation.evaluate('data/processed/eclipse/rank.txt')
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(report)
