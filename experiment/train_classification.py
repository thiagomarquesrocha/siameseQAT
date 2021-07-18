"""
Trains a SiameseQAT model for retrieval experiment using bug reports
preprocessed
"""
import os
import mlflow
import click
import logging
from mlflow.utils.logging_utils import eprint
from src.evaluation.retrieval import Retrieval
from src.utils.keras_utils import KerasUtils
from src.deep_learning.training.train_retrieval import TrainRetrieval
from src.deep_learning.training.train_classification import TrainClassification
from src.deep_learning.training.train_config import TrainConfig
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

logging.basicConfig(level=logging.DEBUG)

@click.command(
    help="SiameseQAT script for retrieval experiment"
)
@click.option("--run_id_retrieval", default='', help='Hash id for retrieval previously runned.')
@click.option("--domain", default='eclipse', help='Dataset to be used. Ex: eclipse, netbeans, openoffice.')
@click.option("--batch_size", default=64, type=int, help="Batch size for training and validation phase.")
@click.option("--epochs", default=15, type=int, help="Number of epochs for training.")
def train_classification(run_id_retrieval, domain, batch_size, epochs):

    with mlflow.start_run(run_id=run_id_retrieval, run_name="retrieval") as active_run:

        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri))

        # Fetch a specific artifact uri
        artifact_uri = mlflow.get_artifact_uri(artifact_path="encoder_model/encoder_model.ckpt")
        artifact_uri = artifact_uri[8:]
        print("Artifact uri: {}".format(artifact_uri))

        retrieval_params = active_run.data.params

        preprocessing = retrieval_params['preprocessing']

        dir_input = os.path.join('data', 'processed', domain, )

        print("Data params:", active_run.data.params)
        print("Data metrics:", active_run.data.metrics)

        # Autolog
        mlflow.keras.autolog()

        # https://stackoverflow.com/questions/60354923/how-can-i-handle-the-variable-uninitialized-error-in-tensorflow-v2
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)

            model_name = retrieval_params['model_name']
            epochs_trained = int(retrieval_params['epochs'])
            title_seq_lenght = int(retrieval_params['title_seq'])
            desc_seq_lenght = int(retrieval_params['desc_seq'])
            bert_layers = int(retrieval_params['bert_layers'])
            batch_size_retrieval = int(retrieval_params['batch_size'])
            dir_input = os.path.join('data', 'processed', domain, preprocessing)
            retrieval = TrainRetrieval(model_name, 
                        dir_input, 
                        domain, 
                        preprocessing, 
                        MAX_SEQUENCE_LENGTH_T=title_seq_lenght, 
                        MAX_SEQUENCE_LENGTH_D=desc_seq_lenght,
                        BERT_LAYERS=bert_layers, 
                        EPOCHS=epochs_trained, 
                        BATCH_SIZE=batch_size_retrieval, 
                        BATCH_SIZE_TEST=batch_size_retrieval).build()

            retrieval_preload = retrieval.get_model()
            
            # Classification
            pretrained_model_input = artifact_uri
            train = TrainClassification(retrieval_preload, 
                        model_name, 
                        pretrained_model_input, 
                        dir_input, 
                        domain, 
                        preprocessing, 
                        EPOCHS=epochs, 
                        BATCH_SIZE=batch_size, 
                        BATCH_SIZE_TEST=batch_size)

            train.run()        

if __name__ == "__main__":
    train_classification()