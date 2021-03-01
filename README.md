## DeepQL : A Semantic Context-Based Duplicate Bug Report Detection using Replicated Cluster Information

In large-scale software development environments, defect reports are maintained through Bug Tracking Systems (BTS) and analyzed by domain experts. Since different users may create bug reports in a non-standard manner, each user can report a particular problem with a unique set of words. Therefore, different reports may describe the same problem, generating duplication. 


 In order to avoid redundant tasks for the development team, an expert needs to look at all new reports while trying to label possible duplicates. However, this approach is neither trivial nor scalable and has a direct impact on bug fix correction time. 
 
 
 Recent efforts to find duplicate bug reports tend to focus on deep neural approaches that consider hybrid information from bug reports as textual and categorical features.
 
 
 Unfortunately, these approaches ignore that a single bug can have multiple previously identified duplicates and, therefore, multiple textual descriptions, titles, and categorical information.
 
 
 In this work, we propose DeepQL, a duplicate bug report detection method that considers not only information on individual bugs, but also collective information from bug clusters. The DeepQL combines context and semantic learning on textual and categorical features, as also topic-based features, with a novel loss function called Quintet Loss, which considers the centroid of duplicate clusters and their contextual information.
 
 
 We validated our approach on the well-known open-source software repositories Eclipse, Netbeans, and Open office, that comprises more than 500 thousand bug reports. We evaluated both retrieval and classification of duplicates, reporting a Recall@25 mean of 71% for retrieval, and 99% AUROC for classification tasks, results that were significantly superior to previous works.

## 1. PREREQUISITES

Some libraries in python environment are required to enable the source code run properly.

**First, install pipenv**

```
$ pip install pipenv
```

**Install dependencies**

```
$ pipenv install
```

**Export the directory ```src/``` to ```PYTHONPATH```**

```
$ export PYTHONPATH=src # Linux
$ set PYTHONPATH=src # Windows
```

**Run tests**

To run all tests you will need BERT pretrained uncased (uncased_L-12_H-768_A-12)[https://github.com/google-research/bert/blob/master/README.md]. Download and unpack on root directory.

```
$ pipenv run pytest tests
```

**Run tests looking the DEBUG level messages**

```
$ pipenv run pytest --log-cli-level=DEBUG tests
```

# Old installation

$ pip install {library}

lybrary:
- scikit-learn
- numpy
- matplotlib
- pickle
- keras
- tqdm
- annoy
- keras_bert
- keras_radam

Jupyter notebook it is required to run the trainings.

Create a file in the root named as "modelos", to save all model trained, and "resultados", to save all model results.

### 2. PREPROCESSING

To run the experiments need to preprocess the datasets and the preprocessing_bert.py.

#### 2.1 - Start using CLI

$ python src/cli.py {dataset} no-colab

The dataset from [Lazar et al. (2014)](http://alazar.people.ysu.edu/msr14data/) has the following open-source software repositories:
- eclipse
- netbeans
- openoffice

After run the 2.1 command the following directories will be created in root directory:

- data/processed/eclipse
- data/processed/openoffice
- data/processed/netbeans

For each directory will be create files from train, test, vocabulary corpus and features categorical from bug reports.

- bugs/ : a list of pickle objects to save a bug report document in json format. All bugs are saved by id. Ex: 1.pkl, 2.pkl, ..., etc.
- train.txt : IDs from bugs that will be used for training
- test.txt : IDs from bugs that will be used for test
- word_vocab.pkl : dictionary list of words present in dataset saved in pickle format.
- bug_ids.txt : IDs from all bugs in the dataset.
- normalized_bugs.json : All bugs reports saved in json format normalized.
- bug_pairs.txt : list of duplicate pairs available by Lazar et. al. 2014.
- bug_severity.dic : dictionary for severities categorical feature.
- bug_status.dic : dictionary for for all bug report status.
- component.dic : dictionary for for all bug report components.
- priority.dic : dictionary for all bug report priorities.
- product.dic : dictionary for all bug report products.


### 3. EXPERIMENTS

#### 3.1 RETRIEVAL EXPERIMENTS ##

To train the models to evaluate in retrieval task the following files need to be executed. All models are trained
on train.txt file and evaluated using test.txt.

- baseline_dms.ipynb - Train the DMS baseline.
- baseline_dwen.ipynb - Train the DWEN baseline.
- deepTL.ipynb - Train the DeepTL.
- deepQL.ipynb - Train the DeepQL baseline.

#### 3.2 CLASSIFICATION EXPERIMENTS

To train the models to evaluate in classification task the following files need to be executed. All models are trained
on train.txt file and evaluated using test.txt.

- classification_baseline_dms.ipynb - Train the DMS.
- classification_baseline_dwen.ipynb -  Train the DWEN.
- classification_propose_QL_and_TL.ipynb - Train the DeepTL and DeepQL.

### 4. RESULTS

The following files are used to support analyse the results.

- result.ipynb - See all results of retrieval and classification in tabular format.
- retrieval.ipynb - Generate the retrieval evaluated separated using the test file for all models.
- textual_analysis.ipynb - Analyse the vocabulary for all datasets for textual features.
- buckets_analysis.ipynb - Plot with TSNE the embedding latent space for set of duplicates.
- dataset_statistics.ipynb - Statistics generated for all datasets.
