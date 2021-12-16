from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as tfb


# From https://developpaper.com/tensorflow-chapter-tensorflow-2-x-local-training-and-evaluation-based-on-keras-model/
class WeightedBinaryCrossEntropy(keras.losses.Loss):
    """
    Args:
    pos_weight: Scalar to affect the positive labels of the loss function.
    weight: Scalar to affect the entirety of the loss function.
    from_logits: Whether to compute loss from logits or the probability.
    reduction: Type of tf.keras.losses.Reduction to apply to loss.
    name: Name of the loss function.
    """

    def __init__(self,
                 pos_weight,
                 weight,
                 from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.losses.binary_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
        )[:, None]
        ce = self.weight * (ce * (1 - y_true) + self.pos_weight * ce * y_true)

        return ce


def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    From https://helioml.org/08/notebook.html
    """
    # Multiplier for positive targets, needs to be tuned
    POS_WEIGHT = 5

    # Transform back to logits
    _epsilon = tf.convert_to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tfb.log(output / (1 - output))

    # Compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)

    return tf.reduce_mean(loss, axis=-1)
    

def split_data(df, yy_train, yy_test, attributes, ylabel):
    """"Split the data into train and test
         df is the data\n",
         attributes are the covariates,
         ylabel is the target variable"""
    train_dataset = df[(df.date.dt.year >= yy_train[0]) &
                       (df.date.dt.year <= yy_train[1])]
    test_dataset = df[(df.date.dt.year >= yy_test[0]) &
                      (df.date.dt.year <= yy_test[1])]

    # Extract the dates for each datasets
    train_dates = train_dataset['date']
    test_dates = test_dataset['date']

    # Extract labels
    train_labels = train_dataset[ylabel].copy()
    test_labels = test_dataset[ylabel].copy()
    
    # Extract predictors\n",
    train_dataset = train_dataset[attributes]
    test_dataset = test_dataset[attributes]

    return(train_dataset, train_labels, test_dataset, test_labels, train_dates, test_dates)


def create_pipeline(data, cat_var):
    """Prepare the data in the right format for the model"""

    num_attribs = list(data)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    if (cat_var != None):
        num_attribs.remove(cat_var)
        cat_attribs = [cat_var]
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
    else:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
        ])

    return(full_pipeline)


def evaluate_model(test_labels, train_labels, predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')


class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, batch_size=32, shuffle=True, load=True, mean=None, std=None):
    #def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, mean=None, std=None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours # I am skipping it for now
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """
        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        #self.lead_time = lead_time

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            #try:
            #    data.append(ds[var].sel(level=levels))
            #except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        # I removed self.data.mean(('time', 'lat', 'lon')).compute()
        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')) if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')) if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        #self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        #self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        #self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        #self.on_epoch_end()
        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    