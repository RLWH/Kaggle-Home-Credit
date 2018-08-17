import os
import json
import argparse
import numpy as np
import xgboost as xgb

from utils import *
from models.XGBModel import generate_params_set, XGBModel

parser = argparse.ArgumentParser(description='Train a set of models by input a dataset, transformation config and training config')
parser.add_argument('-p', '--pipeline', help='File path for the pipeline json config file')
# parser.add_argument('hparams', help='File path for the hyperparameters json config file (Not ready)')
parser.add_argument('-tp', '--tparams', help='File path for the training parameters json config file (Not ready)')

args = parser.parse_args()


def train():

    """
    General training function
    """

    """
    PART 1: DATA CLEANSING AND PREPROCESSING
    """

    # Load the file
    pipeline_path = os.path.join(os.getcwd(), args.pipeline)
    tparams_path = os.path.join(os.getcwd(), args.tparams)

    print("Getting pipeline config file from path %s" % pipeline_path)
    with open(pipeline_path) as f:
        feature_config = json.load(f)

    print("Getting training parameters config file from path %s" % tparams_path)
    with open(tparams_path) as f:
        training_config = json.load(f)

    merged_features, target = load_data(feature_config)
    merged_features = merged_features.fillna(value=0)

    print("Features to be used for training: %r " % merged_features.columns.values)
    print("Target name: %r" % target.name)

    # Split dataset
    X_train, X_val, y_train, y_val = split_data(merged_features, target)

    """
    PART 2. Read the model json and start building the model accordingly
    
    Expect: 
    model = new Model(params)
    model.train()
    model.eval()
    model.predict
    
    """

    """
    PART 3. Choice of ensemble
    """


if __name__ == "__main__":
    train()