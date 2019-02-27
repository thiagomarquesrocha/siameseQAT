import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os

class Normalization():
  def __init__(self, DATASET, FILE):
    print('Starting the normalization...')
    self.RAW_DIR = 'data/raw'
    self.check_directory(DATASET, FILE)

  def check_directory(self, DATASET, FILE):
    # Checking the normalized data
    norm_dir = os.path.join('data/normalized', DATASET)
    if not os.path.exists(norm_dir):
        os.mkdir(norm_dir)
    self.EXPORT = os.path.join(norm_dir, FILE + '.csv')
    # Checking the raw data
    self.RAW_DIR = os.path.join(self.RAW_DIR, DATASET)
    if not os.path.exists(self.RAW_DIR):
      raise Exception('Raw data not available!')
    self.RAW_DIR = os.path.join(self.RAW_DIR, FILE + '.json')

  def run(self):
    print(self.RAW_DIR)
    # Read dataset
    df = pd.read_json(self.RAW_DIR, lines=True)
    # Drop useless columns
    df.drop('_id', inplace=True, axis=1)
    # Export dataset
    print("Exporting {}".format(self.EXPORT))
    df.to_csv(self.EXPORT, index=False)


def main():
    datasets = ['eclipse', 'eclipse', 
                  'eclipse', 'eclipse', 
                  'netbeans', 'netbeans', 
                  'openoffice', 'openoffice']
    files = ['eclipse', 'eclipse_pairs', 
              'eclipse_small', 'eclipse_small_pairs', 
                'netbeans', 'netbeans_pairs', 
                  'openOffice', 'openOffice_pairs']
    for f, d in zip(files, datasets):
      normalization = Normalization(DATASET=d, FILE=f)
      normalization.run()

if __name__ == '__main__':
  main()