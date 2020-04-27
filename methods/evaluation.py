import numpy as np
import pprint
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class Evaluation():
    def __init__(self, verbose=1):
        self.verbose = verbose
        self.MAX_RANK = 25
    
    def verbose(self, verbose=1):
        self.verbose = verbose

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax, fig

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
        self.recall_at_1_corrects_sum,\
        self.recall_at_5_corrects_sum, self.recall_at_10_corrects_sum, \
        self.recall_at_15_corrects_sum, self.recall_at_20_corrects_sum, self.recall_at_25_corrects_sum = 0, 0, 0, 0, 0, 0
        self.recall_at_1_total_sum, \
        self.recall_at_5_total_sum, self.recall_at_10_total_sum, self.recall_at_15_total_sum, \
        self.recall_at_20_total_sum, self.recall_at_25_total_sum = 0, 0, 0, 0, 0, 0
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
            '0 - recall_at_1' : round(self.recall_at_1_corrects_sum / self.recall_at_1_total_sum, 2),
            '1 - recall_at_5' : round(self.recall_at_5_corrects_sum / self.recall_at_5_total_sum, 2),
            '2 - recall_at_10' : round(self.recall_at_10_corrects_sum / self.recall_at_10_total_sum, 2),
            '3 - recall_at_15' : round(self.recall_at_15_corrects_sum / self.recall_at_15_total_sum, 2),
            '4 - recall_at_20' : round(self.recall_at_20_corrects_sum / self.recall_at_20_total_sum, 2),
            '5 - recall_at_25' : round(self.recall_at_25_corrects_sum / self.recall_at_25_total_sum, 2)
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

class EvaluationPrecision(Evaluation):
        def top_k_recall(self, row, k):
            query, rank = row.split('|')
            query_dup_id, ground_truth = query.split(":")
            ground_truth = np.asarray(ground_truth.split(','), int)
            candidates = [int(item.split(':')[0]) for pos, item in enumerate(rank.split(",")[:self.MAX_RANK])]
            corrects = len(set(ground_truth) & set(candidates[:k]))
            corrects = corrects / k
            total = 1
            return float(corrects), total

class EvaluationRecall(Evaluation):
        def top_k_recall(self, row, k):
            query, rank = row.split('|')
            query_dup_id, ground_truth = query.split(":")
            ground_truth = np.asarray(ground_truth.split(','), int)
            candidates = [int(item.split(':')[0]) for pos, item in enumerate(rank.split(",")[:self.MAX_RANK])]
            corrects = len(set(ground_truth) & set(candidates[:k]))
            relevant_queries = k if(len(ground_truth) > k) else len(ground_truth)
            corrects = corrects / relevant_queries
            total = 1
            return float(corrects), total

class EvaluationFscore():
        def calculate(self, precision, recall):
            return 2 * precision * recall / (precision + recall)
        def evaluate(self, precision, recall):
            report = {
                '0 - recall_at_1' : round(self.calculate(precision['0 - recall_at_1'], recall['0 - recall_at_1']), 2),
                '1 - recall_at_5' : round(self.calculate(precision['1 - recall_at_5'], recall['1 - recall_at_5']), 2),
                '2 - recall_at_10' : round(self.calculate(precision['2 - recall_at_10'], recall['2 - recall_at_10']), 2),
                '3 - recall_at_15' : round(self.calculate(precision['3 - recall_at_15'], recall['3 - recall_at_15']), 2),
                '4 - recall_at_20' : round(self.calculate(precision['4 - recall_at_20'], recall['4 - recall_at_20']), 2),
                '5 - recall_at_25' : round(self.calculate(precision['5 - recall_at_25'], recall['5 - recall_at_25']), 2)
            }
            return report

if __name__ == '__main__':
    evaluation = Evaluation()
    report = evaluation.evaluate('data/processed/eclipse/rank.txt')
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(report)
