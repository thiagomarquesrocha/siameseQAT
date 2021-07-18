## SiameseQAT : A Semantic Context-Based Duplicate Bug Report Detection using Replicated Cluster Information

**Paper**: https://ieeexplore.ieee.org/document/9380447

**Abstract**:

In large-scale software development environments, defect reports are maintained through bug tracking systems (BTS) and analyzed by domain experts. Different users may create bug reports in a nonstandard manner, and may report a particular problem using a particular set of words due to stylistic choices
and writing patterns. 

Therefore, the same defect can be reported with very different descriptions, generating non-trivial duplicates. To avoid redundant work for the development team, an expert needs to look at all new reports while trying to label possible duplicates. However, this approach is neither trivial nor scalable and directly impacts on bug fix correction time. Recent efforts to find duplicate bug reports tend to focus on deep neural approaches that consider hybrid representations of bug reports, using both structured and unstructured information. Unfortunately, these approaches ignore that a single bug can have multiple previously identified
duplicates and, therefore, multiple textual descriptions, titles, and categorical information. 

In this work, we propose **SiameseQAT**, a duplicate bug report detection method that considers information on individual bugs as well as information extracted from bug clusters. The SiameseQAT combines context and semantic learning on structured and unstructured features and corpus topic extraction-based features, with a novel loss function called Quintet Loss, which considers the centroid of duplicate clusters and their contextual information. We validated our approach on the well-known open-source software repositories **Eclipse, NetBeans, and Open Office**, comprised of more than **500 thousand bug reports**. We evaluated both the
retrieval and classification of duplicates, reporting a Recall@25 mean of **85% for retrieval** and **84% AUROC for classification** tasks, results that were significantly superior to previous works.

![SiameseQAT](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6287639/9312710/9380447/rocha4-3066283-small.gif)

![SiameseQAT](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6287639/9312710/9380447/rocha5-3066283-small.gif)


## 1. PREREQUISITES

Some libraries in python environment are required to enable the source code run properly.

**Download Dataset**

[dataset.zip](https://drive.google.com/file/d/1reRGkmSItk0MJyiefbIjEAEfujAg7JDk/view?usp=sharing)

```
# Create on root directory /data
-> mkdir /data
# Unzip on root directory /data
-> unzip dataset.zip
# See on data/normalized/
# - eclipse
# - openoffice
# - netbeans
```

**First, install pipenv**

```
$ pip install pipenv
```

**Install dependencies**

```
$ pipenv install
```

**Export the root directory ```.``` to ```PYTHONPATH```**

```
$ export PYTHONPATH=. # Linux
$ set PYTHONPATH=. # Windows
```

**Run tests**

To run all tests you will need BERT pretrained [uncased_L-12_H-768_A-12](https://github.com/google-research/bert/blob/master/README.md). Download and unpack on root directory.

```
$ pipenv run pytest tests
```

**Run tests looking the DEBUG level messages**

```
$ pipenv run pytest --log-cli-level=DEBUG tests
```

### 2. WORKFLOW

*Image will be done to illustrate the workflow*

#### 2.1 PREPROCESSING

To run the experiments need to preprocess the datasets and the cli.py.

##### Start using CLI

```
$ python src/cli.py {dataset} no-colab 
```

The dataset from [Lazar et al. (2014)](http://alazar.people.ysu.edu/msr14data/) has the following open-source software repositories:

##### {dataset}

- eclipse
- netbeans
- openoffice

###### {preprocessing}

- bert
- baseline

After run all previously steps the following directories will be created in root directory:

- data/processed/eclipse
- data/processed/openoffice
- data/processed/netbeans

For each directory will be create files to train, test, vocabulary corpus and categorical features from bug reports.

- **bugs/** : a list of pickle objects to save a bug report document in json format. All bugs are saved by id. Ex: 1.pkl, 2.pkl, ..., etc.
- **train_chronological.txt** : IDs from bugs that will be used for training
- **test_choronological.txt** : IDs from bugs that will be used for test
- **word_vocab_bert.pkl** : dictionary list of words present in dataset saved in pickle format.
- **bug_ids.txt** : IDs from all bugs in the dataset.
- **normalized_bugs.json** : All bugs reports saved in json format normalized.
- **bug_pairs.txt** : list of duplicate pairs available by Lazar et. al. 2014.
- **bug_severity.dic** : dictionary for severities categorical feature.
- **bug_status.dic** : dictionary for for all bug report status.
- **component.dic** : dictionary for for all bug report components.
- **priority.dic** : dictionary for all bug report priorities.
- **product.dic** : dictionary for all bug report products.


#### 2.2 EXPERIMENTS

##### RETRIEVAL EXPERIMENTS ##

To train the model to evaluate in retrieval task run the following command. All models are trained on train.txt file and evaluated using test.txt.

- Model available:
    - SiameseTA
    - SiameseTAT
    - SiameseQAT-A
    - SiameseQAT-W
    - SiameseQA-A
    - SiameseQA-W

- Parameters available:
    - **model_name**: Model name to be used. Ex: SiameseQA-A, SiameseQAT-W, SiameseTA
    - **domain**: Dataset to be used. Ex: eclipse, netbeans, openoffice.
    - **title_seq**: Title length sequence to be used in model.
    - **desc_seq**: Description length sequence to be used in model.
    - **batch_size**: Batch size for training and validation phase.
    - **epochs**: Number of epochs for training.
    - **bert_layers**: Number of bert unfrozen layers for training.
    - **preprocessing**: Type of preprocessing for models. Ex: bert, keras

**Example of how to run retrieval experiment**


```
mlflow run . --experiment-name retrieval -e train_retrieval -P model_name=SiameseTA -P domain=eclipse_test -P title_seq=1 -P desc_seq=1 -P batch_size=1 -P bert_layers=1
```

##### CLASSIFICATION EXPERIMENTS

To train the model to evaluate in classification task run the following command. Note that all models are trained
on train.txt file and evaluated using test.txt.

**Example of how to run classification experiment**

Note that run_id_retrieval has a already valid id.

```
mlflow run . --experiment-name classification -e train_classification -P run_id_retrieval=66f2b01699474634bd9e6559244c4d26 -P domain=eclipse_test -P batch_size=3 -P
epochs=1
```

#### 2.3 RESULTS

The following files are used to support analyse the results.


**TBD**: Refactoring in development to use mlflow 
