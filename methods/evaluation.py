import numpy as np

class Evaluation():
    def __init__(self):
        pass
    
    """
        Rank recall_rate_@k
        rank = "query:master|master:id:sim,master:id:sim"
    """
    def top_k_recall(self, rank, k):
        query, rank = rank.split('|')
        query_dup_id, query_master = query.split(":")
        query_master = int(query_master)
        hit = 0
        for pos, item in enumerate(rank.split(",")):
            master, dup, sim = item.split(':')
            dup = int(dup)
            master = int(master)
            if master == query_master and (pos+1) <= k:
                hit=1
                return [hit]
        return [hit]

    def evaluate(self, path):
        recall_at_5, recall_at_10, recall_at_15, recall_at_20 = [], [], [], []
        total = 0
        with open(path, 'r') as file_input:
            for row in file_input:
                if row == '': continue
                recall_at_5 += self.top_k_recall(row, k=5)
                recall_at_10 += self.top_k_recall(row, k=10)
                recall_at_15 += self.top_k_recall(row, k=15)
                recall_at_20 += self.top_k_recall(row, k=20)
                total+=1

        report = {
            'recall_at_5' : round(sum(recall_at_5) / total, 2),
            'recall_at_10' : round(sum(recall_at_10) / total, 2),
            'recall_at_15' : round(sum(recall_at_15) / total, 2),
            'recall_at_20' : round(sum(recall_at_20) / total, 2)
        }

        return report

if __name__ == '__main__':
    evaluation = Evaluation()
    report = evaluation.evaluate('data/processed/eclipse/rank.txt')

    print(report)
