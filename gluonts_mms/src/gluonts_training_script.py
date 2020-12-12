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
    

def _get_local_dataset(dataset_location=None):
    """
    This method returns dataset stored locally by Sagemaker. 
    If no param dataset_location is not provided, it will download a new dataset using GluonTS native method.
    """    
    if dataset_location is None: #if S3 not provided, downloading data locally on Sagemaker nodes.
        dataset = get_dataset("electricity", regenerate=False)
    
    else:  #if S3 bucket is specified in os.environ['SM_CHANNEL_TRAINING'], that means that Sagemaker automatically download dataset locally. So no need to download it again.
        logger.info("Attempting to get GluonTS dataset from {}".format(dataset_location))
        root_dir = os.path.join(dataset_location, "data")
        train_dir = os.path.join(root_dir, "train")
        test_dir = os.path.join(root_dir, "test")

        dataset = common.load_datasets(
            metadata=root_dir,
            train=train_dir,
            test=test_dir)
        
    logger.info("GluonTS dataset retrieved successfully...")
        
    return dataset
        


def train(args):
    """
    Main training method. Takes training parameters from Argparser and performs training using DeepAR estimator.
    """
    
    dataset = _get_local_dataset(os.environ['SM_CHANNEL_TRAINING'])
    freq = dataset.metadata.freq if args.freq==None else args.freq # If user doesn't specify frequency, then default frequency is used for given dataset.
    cardinality = [int(dataset.metadata.feat_static_cat[0].cardinality)] if dataset.metadata.feat_static_cat!=[] and args.use_static_features else None
    prediction_lenght = dataset.metadata.prediction_length # default prediction lenght for dataset is used
    
    device = mx.context.gpu() if mx.context.num_gpus()>0 else mx.context.cpu() # if GPU device is available, then use it. Otherwise, CPU.
    
    trainer = Trainer(ctx=device, 
                      epochs=args.epochs, 
                      hybridize=True,
                      num_batches_per_epoch=args.batches_per_epoch)
    
    estimator = DeepAREstimator(freq=freq, 
                                prediction_length=prediction_lenght, # recommended by electricity dataset.
                                context_length=args.context_length, # default for model
                                num_layers=args.num_layers, # default for model
                                num_cells=args.num_cells, #default for model
                                cell_type='lstm', #default for model
                                use_feat_static_cat = args.use_static_features, # as dataset has static feature, let's use them.
                                cardinality=cardinality, # recommended by electricity dataset
                                embedding_dimension=[50], # default value is [min(50, (cat + 1) // 2) for cat in self.cardinality
                                scaling=True, # default for model
                                dropout_rate=args.dropout, # default for model
                                trainer=trainer
                                )
        
    predictor = estimator.train(training_data=dataset.train)
    return predictor

def evaluate(predictor, model_type):
    dataset = _get_local_dataset(os.environ['SM_CHANNEL_TRAINING'])
    forecast_it, ts_it = make_evaluation_predictions(dataset.test, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    _plot_forecasts(tss, forecasts, model_type, past_length=150, num_plots=3)
    evaluator = Evaluator(quantiles=[0.5])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))
    logger.info(agg_metrics)


def _plot_forecasts(tss, forecasts, model_type, past_length, num_plots):
    
    import matplotlib.pyplot as plt
    
    random_id = randint(0,1000)
    plot_id = 1

    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.title('Evaluation results for {} model'.format(model_type))
        plt.show()
        plt.savefig(os.environ['SM_OUTPUT_DATA_DIR']+'/chart_{}-{}.png'.format(plot_id, random_id))
        plot_id += 1
    
def save_trained_model(predictor):
    """
    Serializes GluonTS predictor and saves in Sagemaker model directory (defined in 'SM_MODEL_DIR' environmental variable).
    Sagemaker will automatically upload model artifacts at the end of training job.
    """
    predictor.serialize(Path(os.environ['SM_MODEL_DIR']))

    
def model_fn(model_dir):
    """
    Loads trained model and returns GluonTS predictor.
    """
    logger.info(f"Loading model from {model_dir}")
    predictor_deserialized = Predictor.deserialize(Path(model_dir))
    
    return predictor_deserialized


def transform_fn(model, data, content_type, output_content_type):
    """
    Parses request for prediction and returns predictions, where:
        - model is the model objected loaded by model_fn, 
        - request_body is the data from the inference request, 
        - content_type is the content type of the request, 
        - accept_type is the request content type for the response.
        
    Returns on of the following:
        - a tuple with two items: the response data and accept_type (the content type of the response data), OR
        - the response data: (the content type of the response is set to either the accept header in the initial request or default to “application/json”)
    """
    
    deser_data = pd.read_json(data, typ='series')
    test_data = ListDataset([{"start": deser_data.index[0],
                              "target": deser_data.values,
                              "feat_static_cat":[0]}], # Defining feat_static_cat seems like stub because of GluonTS underlying API issues.
                            freq="1H")
    
    forecast_it, ts_it = make_evaluation_predictions(test_data, model,  num_samples=100)
    response_body = json.dumps({'predictions':list(forecast_it)[0].samples.tolist()[0]})
    
    logger.info("Completed predictions successfully.")
    
    return response_body, output_content_type
    
    
if __name__ == "__main__":
    """
    At training time, Sagemaker executes code below.
    At inference time, Sagemaker calls reserved functions such as transform_fn and model_fn.
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--freq', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batches-per-epoch', type=int, default=100)
    parser.add_argument('--prediction-length', type=int, default=24)
    parser.add_argument('--context-length', type=int, default=24)
    parser.add_argument('--use-static-features', type=bool, default=True)
    parser.add_argument('--cardinality', type=int, default=321)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-cells', type=int, default=40)
    
    args = parser.parse_args()
    
    
    # Code below is executed at training time.
    predictor = train(args)
    evaluate(predictor, "deepar")
    save_trained_model(predictor)
    
    
    

    
    
    
    
        