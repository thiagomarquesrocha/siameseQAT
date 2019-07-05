import numpy as np
import pprint

class Evaluation():
    def __init__(self, verbose=1):
        self.verbose = verbose
    
    """
        Rank recall_rate_@k
        rank = "query:master|master:id:sim,master:id:sim"
    """
    def top_k_recall(self, rank, k):
        query, rank = rank.split('|')
        query_dup_id, query_master = query.split(":")
        query_master = int(query_master)
        rank_masters = [int(item.split(':')[0]) for pos, item in enumerate(rank.split(",")[:25])]
        corrects = len(set([query_master]) & set(rank_masters[:k]))
        #total = len(retrieval.buckets[issues_by_buckets[query_master]])
        total = 1
        #total = 1 if corrects <= 0 else corrects
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
            'recall_at_5' : round(self.recall_at_5_corrects_sum / self.recall_at_5_total_sum, 2),
            'recall_at_10' : round(self.recall_at_10_corrects_sum / self.recall_at_10_total_sum, 2),
            'recall_at_15' : round(self.recall_at_15_corrects_sum / self.recall_at_15_total_sum, 2),
            'recall_at_20' : round(self.recall_at_20_corrects_sum / self.recall_at_20_total_sum, 2),
            'recall_at_25' : round(self.recall_at_25_corrects_sum / self.recall_at_25_total_sum, 2)
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
