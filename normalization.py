import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os

class Normalization():
  def __init__(self, DATASET, FILE, BUCKET):
    self.RAW_DIR = 'data/raw'
    self.check_directory(DATASET, FILE, BUCKET)

  def check_directory(self, DATASET, FILE, BUCKET):
    # Checking the normalized data
    norm_dir = os.path.join('data/normalized', DATASET)
    if not os.path.exists(norm_dir):
        os.mkdir(norm_dir)
    self.EXPORT = os.path.join(norm_dir, FILE + '.csv')
    self.EXPORT_PAIRS = os.path.join(norm_dir, BUCKET + '.csv')
    # Checking the raw data
    self.RAW_DIR = os.path.join(self.RAW_DIR, DATASET)
    if not os.path.exists(self.RAW_DIR):
      raise Exception('Raw data not available!')
    self.RAW_PAIRS_DIR = os.path.join(self.RAW_DIR, BUCKET + '.json')
    self.RAW_DIR = os.path.join(self.RAW_DIR, FILE + '.json')

  def exporting_dataset(self):
    print("Starting the normalization: {}".format(self.RAW_DIR))
    # Read dataset
    df = pd.read_json(self.RAW_DIR, lines=True)
    # Drop useless columns
    df.drop('_id', inplace=True, axis=1)
    # Export dataset
    print("Exporting {}".format(self.EXPORT))
    df.to_csv(self.EXPORT, index=False)
  def exporting_pairs(self):
    print("Starting the normalization: {}".format(self.RAW_PAIRS_DIR))
    # Read dataset
    df = pd.read_json(self.RAW_PAIRS_DIR, lines=True)
    # Drop useless columns
    df.drop('_id', inplace=True, axis=1)

    # Creating the buckets
    buckets = {}
    progress = tqdm(total=df.shape[0])
    for row in df.iterrows():
        name, dup, duplicated = row[1]
        if not name in buckets:
          buckets[name] = []
        if duplicated == 1:
          buckets[name].append(str(dup))
        progress.update(1)
    progress.close()

    progress = tqdm(total=len(buckets))
    for key in buckets:
        row = buckets[key]
        removing_dup_ids = set(row)
        dups = ";".join(list(removing_dup_ids))
        buckets[key] = dups if dups != '' else np.nan
        progress.update(1)
    progress.close()

    buckets = [(key, buckets[key]) for key in buckets]
    df = pd.DataFrame(buckets, columns=['issue_id', 'duplicate'])
    df.to_csv(self.EXPORT_PAIRS, index=False)
    print("Exported {}".format(self.EXPORT_PAIRS))

  def run(self):
    self.exporting_dataset()
    self.exporting_pairs()

def main():
    datasets = ['eclipse', 'eclipse', 'netbeans', 'openoffice']
    issues = ['eclipse', 'eclipse_small', 'netbeans', 'openOffice']
    buckets = ['eclipse_pairs',  'eclipse_small_pairs', 'netbeans_pairs', 'openOffice_pairs']
    for issue, dataset, bucket in zip(issues, datasets, buckets):
      normalization = Normalization(DATASET=dataset, FILE=issue, BUCKET=bucket)
      normalization.run()

if __name__ == '__main__':
  main()