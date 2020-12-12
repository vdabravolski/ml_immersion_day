# Standard library imports
from pathlib import Path
import argparse
import os
import json
import logging
from itertools import islice
from random import randint
from pathlib import Path
import pandas as pd


# MXNet & GluonTS imports
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.util import to_pandas
import mxnet as mx
from gluonts.dataset import common
from gluonts.dataset.repository import datasets
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset


# Logging: print logs analogously to Sagemaker.
logger = logging.getLogger(__name__)

class ModelHandler(object):
    """
    Keras VGG pre-trained model classifier
    """

    def __init__(self):
        self.initialized = False
        self.model = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return: None
        """
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir") 
        
        logger.info(f"Loading model from {model_dir}")
        self.model = Predictor.deserialize(Path(model_dir))
       

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """

        deser_data = pd.Series(request[0]["body"])
        deser_data.index = [i[:18] for i in deser_data.index] # stripping timezone info
    
        test_data = ListDataset([{"start": deser_data.index[0],
                                  "target": deser_data.values,
                                  "feat_static_cat":[0]}], # Defining feat_static_cat seems like stub because of GluonTS underlying API issues.
                                freq="1H")

        return test_data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in numpy array
        """
        forecast_it, ts_it = make_evaluation_predictions(model_input, self.model,  num_samples=100)
        
        return forecast_it

    def postprocess(self, forecast_it):
        """
        Post processing step - converts predictions to str
        :param inference_output: predictions as numpy
        :return: list of inference output as string
        """
        response_body = json.dumps({'predictions':list(forecast_it)[0].samples.tolist()[0]})
        logger.info("Completed predictions successfully.")
    
        return [response_body]

        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        out = self.postprocess(model_out)
        
        return out
    
_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)