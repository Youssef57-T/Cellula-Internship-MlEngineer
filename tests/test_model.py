"""Checks on the model files that are committed to the repository and served by the app."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from conftest import GRAND_CENTRAL, JFK, ROOT, TIMES_SQUARE
from train import load_config


@pytest.fixture(scope='module')
def pipeline():
    return joblib.load(ROOT / 'uber_fare_model.pkl')


@pytest.fixture(scope='module')
def metadata():
    return json.loads((ROOT / 'uber_fare_metadata.json').read_text())


@pytest.fixture(scope='module')
def metrics():
    return json.loads((ROOT / 'metrics.json').read_text())


def predict(pipeline, metadata, pickup, dropoff, when='2014-06-12 18:30:00'):
    row = {'pickup_datetime': when,
           'pickup_latitude': np.radians(pickup[0]), 'pickup_longitude': np.radians(pickup[1]),
           'dropoff_latitude': np.radians(dropoff[0]), 'dropoff_longitude': np.radians(dropoff[1]),
           'passenger_count': 1, 'Car Condition': 'Good', 'Weather': 'sunny', 'Traffic Condition': 'Flow Traffic'}
    return float(pipeline.predict(pd.DataFrame([row])[metadata['raw_features']])[0])


def test_model_was_saved_with_the_installed_scikit_learn(metadata):
    assert metadata['sklearn_version'] == sklearn.__version__


def test_committed_model_passes_the_quality_gate(metrics):
    limit = load_config()['quality_gate']['max_test_mae']
    assert metrics['quality_gate']['passed']
    assert metrics['test']['MAE'] <= limit
    assert metrics['test']['MAE'] < metrics['baseline_median_MAE'] / 2


def test_metadata_and_metrics_describe_the_same_model(metadata, metrics):
    assert metadata['test_metrics']['MAE'] == pytest.approx(metrics['test']['MAE'])
    assert metadata['trained_on_rows'] == metrics['rows']['train']


def test_sanity_trips_are_in_a_plausible_range(pipeline, metadata):
    short = predict(pipeline, metadata, TIMES_SQUARE, GRAND_CENTRAL)   # under 1 km across Midtown
    airport = predict(pipeline, metadata, TIMES_SQUARE, JFK)           # flat fare of $52 plus tolls in 2014
    assert 4 <= short <= 11
    assert 45 <= airport <= 70


def test_longer_trips_cost_more(pipeline, metadata):
    assert predict(pipeline, metadata, TIMES_SQUARE, JFK) > predict(pipeline, metadata, TIMES_SQUARE, GRAND_CENTRAL)


def test_fares_are_higher_after_the_2012_increase(pipeline, metadata):
    before = predict(pipeline, metadata, TIMES_SQUARE, GRAND_CENTRAL, '2011-06-16 18:30:00')
    after = predict(pipeline, metadata, TIMES_SQUARE, GRAND_CENTRAL, '2014-06-12 18:30:00')
    assert after > before
