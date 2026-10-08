"""The training script, run end to end on a 5,000-trip sample of the dataset."""
import json

import joblib
import pandas as pd

import train
from conftest import SAMPLE_DATA


def test_cleaning_rules_remove_invalid_trips():
    config = train.load_config()
    raw = train.load_data(SAMPLE_DATA, config['data_url'])
    clean, removed = train.clean_trips(raw, config['cleaning'])

    assert len(clean) == len(raw) - sum(removed.values())
    assert clean['passenger_count'].between(1, 6).all()
    assert (clean['fare_amount'] >= config['cleaning']['min_fare']).all()
    assert (clean['distance'] > 0).all()
    assert not clean.isna().any().any()


def test_training_saves_a_loadable_model(tmp_path):
    passed, metrics = train.train(SAMPLE_DATA, tmp_path, max_mae=5.0)

    assert passed
    assert metrics['test']['MAE'] < metrics['baseline_median_MAE']
    for name in (train.MODEL_FILE, train.META_FILE, train.METRICS_FILE):
        assert (tmp_path / name).exists()

    pipeline = joblib.load(tmp_path / train.MODEL_FILE)
    metadata = json.loads((tmp_path / train.META_FILE).read_text())
    sample = pd.read_csv(SAMPLE_DATA).dropna().head(5)[metadata['raw_features']]
    assert (pipeline.predict(sample) > 0).all()

    # The comparison models are saved next to it and listed in the metadata the app reads.
    assert len(metadata['other_models']) == 7
    for entry in metadata['other_models']:
        other = joblib.load(tmp_path / entry['file'])
        assert (other.predict(sample) > 0).all()
        assert entry['test_metrics'] == metrics['other_models'][entry['name']]


def test_quality_gate_blocks_a_model_that_is_not_good_enough(tmp_path):
    passed, metrics = train.train(SAMPLE_DATA, tmp_path, max_mae=0.01)

    assert not passed
    assert not metrics['quality_gate']['passed']
    assert not (tmp_path / train.MODEL_FILE).exists()
    assert not (tmp_path / train.OTHER_MODELS_DIR).exists()


def test_command_line_exit_code_follows_the_gate(tmp_path):
    assert train.main(['--data', str(SAMPLE_DATA), '--output-dir', str(tmp_path), '--max-mae', '5']) == 0
    assert train.main(['--data', str(SAMPLE_DATA), '--output-dir', str(tmp_path / 'blocked'), '--max-mae', '0.01']) == 1
