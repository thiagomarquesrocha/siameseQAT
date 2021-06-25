"""
Trains a SiameseQAT model for retrieval experiment using bug reports
preprocessed
"""
import os
import mlflow
import click
import tempfile
from mlflow.utils.logging_utils import eprint
from src.evaluation.retrieval import Retrieval
from src.utils.keras_utils import KerasUtils
from src.deep_learning.training.train_retrieval import TrainRetrieval
from src.deep_learning.training.train_config import TrainConfig

@click.command(
    help="SiameseQAT script for retrieval experiment"
)
@click.option("--model_name", default='SiameseQA-A', help='Model name to be used. Ex: SiameseQA-A, SiameseQAT-W, SiameseTA')
@click.option("--domain", default='eclipse', help='Dataset to be used. Ex: eclipse, netbeans, openoffice.')
@click.option("--title_seq", default=30, type=int, help="Title length sequence to be used in model.")
@click.option("--desc_seq", default=150, type=int, help="Description length sequence to be used in model.")
@click.option("--batch_size", default=64, type=int, help="Batch size for training and validation phase.")
@click.option("--epochs", default=15, type=int, help="Number of epochs for training.")
@click.option("--bert_layers", default=4, type=int, help="Number of bert unfrozen layers for training.")
@click.option("--preprocessing", default='bert', help="Type of preprocessing for models. Ex: bert, keras")
def train_retrieval(model_name, domain, title_seq, 
                desc_seq, batch_size, epochs, bert_layers,
                preprocessing):

    with mlflow.start_run() as active_run:
        dir_input = os.path.join('data', 'processed', domain, preprocessing)

        # Save parameters
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('domain', domain)
        mlflow.log_param('title_seq', title_seq)
        mlflow.log_param('desc_seq', desc_seq)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('bert_layers', bert_layers)
        mlflow.log_param('preprocessing', preprocessing)

        # Autolog
        mlflow.keras.autolog()

        # Train retrieval model
        train = TrainRetrieval(model_name, 
                                dir_input, 
                                domain, 
                                preprocessing, 
                                MAX_SEQUENCE_LENGTH_T=title_seq, 
                                MAX_SEQUENCE_LENGTH_D=desc_seq,
                                BERT_LAYERS=bert_layers, 
                                EPOCHS=epochs, 
                                BATCH_SIZE=batch_size, 
                                BATCH_SIZE_TEST=batch_size).run()
        # Get the model trained
        model = train.get_model()
        encoder = train.get_bug_encoder()

        data = train.train_preparation.get_data()
        # Categorical info
        info_dict = data.info_dict
        verbose = True
        retrieval = Retrieval(domain, info_dict, verbose)
        # Evaluation
        buckets = data.buckets
        train = data.train_data
        test = data.test_data
        bug_set = data.bug_set
        issues_by_buckets = data.issues_by_buckets
        bug_ids = data.bug_test_ids
        method = 'bert'
        only_buckets = False # Include all dups
        
        recall_at_25, exported_rank, _ = retrieval.evaluate(buckets, 
                                                          test, 
                                                          bug_set, 
                                                          encoder, 
                                                          issues_by_buckets, 
                                                          bug_ids, 
                                                          method=method, 
                                                          only_buckets=only_buckets)

        # Save result
        eprint("Saving parameters and metrics")
        mlflow.log_param("train_pair_size", len(train))
        mlflow.log_param("test_pair_size", len(test))
        mlflow.log_param("buckets", len(buckets))

        mlflow.log_metric("recall_at_25", recall_at_25)
        mlflow.log_dict(exported_rank, "exported_rank")

        # Save model
        eprint("Saving model")
        save_model(model, "retrieval_model")
        eprint("Saving encoder")
        save_model(encoder, "encoder_model")

def create_or_get_experiment(client, experiment_name):
    # Case-sensitive name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        exp_id = mlflow.create_experiment(experiment_name)
    # Set an experiment name, which must be unique and case sensitive.
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # Show experiment info
    print("Name: {}".format(experiment.name))
    print("Experiment ID: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    return experiment.experiment_id

def save_model(model, name):
    model_dir = tempfile.mkdtemp()
    model_output = os.path.join(model_dir, "{}.ckpt".format(name))
    KerasUtils.save_weights(model, model_output)
    mlflow.log_artifact(model_output)
    os.remove(model_output)

if __name__ == "__main__":
    train_retrieval()