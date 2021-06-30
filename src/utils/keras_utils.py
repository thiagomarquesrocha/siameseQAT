from tensorflow import keras
import tensorflow as tf
import os
import logging
from src.deep_learning.model.model_base import ModelBase
from src.deep_learning.model.mlp_model import MLPModel
from src.deep_learning.model.bert_model import BERTModel
from src.deep_learning.model.siamese_model import SiameseModel
from src.deep_learning.training.train_config import TrainConfig
from src.deep_learning.loss.quintet_loss import quintet_loss, QuintetWeights, quintet_trainable
from keras_bert import get_custom_objects
from tensorflow.keras.models import load_model as keras_load_model
from src.utils.util import Util
import mlflow
import tempfile

logger = logging.getLogger('KerasUtils')

class KerasUtils:

    @staticmethod
    def log_model(model, name):
        model_dir = tempfile.mkdtemp()
        model_dir = os.path.join(model_dir, name)
        Util.create_dir(model_dir)
        model_output = os.path.join(model_dir, "{}.ckpt".format(name))
        # model_json = model.to_json()
        # model_config = '{}.json'.format(name)
        # model_json_output = os.path.join(model_dir, model_config)
        # with open(model_config, "w") as json_file:
        #     json_file.write(model_json)
        KerasUtils.save_weights(model, model_output)
        mlflow.log_artifact(model_dir)
        #custom_objects = get_custom_objects()
        #custom_objects.update({"ModelBase" : ModelBase, "MLPModel" : MLPModel, "BERTModel" : BERTModel, "SiameseModel" : SiameseModel, "quintet_loss" : quintet_loss, "QuintetWeights" : QuintetWeights, "quintet_trainable" : quintet_trainable})
        #mlflow.keras.log_model(model, name, custom_objects=custom_objects)

    @staticmethod
    def save_weights(model, path):
        """
            See https://www.tensorflow.org/tutorials/keras/save_and_load?hl=en
        """
        m_dir = os.path.join(TrainConfig.OUTPUT_MODELS)
        if not os.path.exists(m_dir):
            os.mkdir(m_dir)
        export = os.path.join(m_dir, path)
        model.save_weights(export)
        logger.debug("Model saved {}".format(export))
    
    @staticmethod
    def load_weights(path, model):
        """
            See https://www.tensorflow.org/tutorials/keras/save_and_load?hl=en
        """
        logger.debug("Loading model {}".format(path))
        model.load_weights(path)

    @staticmethod
    def load_model(path, model):
        """
            See https://keras.io/guides/serialization_and_saving/
            https://www.tensorflow.org/guide/keras/save_and_serialize
        """
        if model == 'SiameseQAT-A' or model == 'SiameseQAT-W' or model == 'SiameseQA-A' or model == 'SiameseQA-W' or model == 'SiameseTAT' or model == 'SiameseTA':
            custom_objects = get_custom_objects()
            custom_objects.update(KerasUtils.get_basic_config())
            return keras_load_model(path, custom_objects=custom_objects)
        return 
    
    @staticmethod
    def save_model(model, export, custom_objects={}, verbose=False):
        """
            See https://keras.io/guides/serialization_and_saving/
            https://www.tensorflow.org/guide/keras/save_and_serialize
        """
        #m_dir = os.path.join(TrainConfig.OUTPUT_MODELS)
        #if not os.path.exists(m_dir):
        #    os.mkdir(m_dir)
        #export = os.path.join(m_dir, path)
        custom_objects.update(KerasUtils.get_basic_config())
        custom_objects.update(get_custom_objects())
        with keras.utils.custom_object_scope(custom_objects):
            model.save(export)
            if(verbose):
                logger.debug("Saved model '{}' to disk".format(export))

    @staticmethod
    def get_basic_config():
        return {"ModelBase" : ModelBase, 
                "MLPModel" : MLPModel, 
                "BERTModel" : BERTModel, 
                "SiameseModel" : SiameseModel, 
                "quintet_loss" : quintet_loss,
                "QuintetWeights" : QuintetWeights,
                "quintet_trainable" : quintet_trainable
                } 
