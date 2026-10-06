"""Train the fare model from the command line and save it only if it passes the quality gate.

This reproduces the final model of Task_2_v2.ipynb (same cleaning rules, split, features and settings),
so the deployed model can be rebuilt without opening the notebook.

    python train.py                      # full data, gate from config.json
    python train.py --data sample.csv --output-dir out --max-mae 5
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from trip_features import CAT_FEATURES, NUM_FEATURES, RAW_FEATURES, add_trip_features

BASE_DIR = Path(__file__).resolve().parent
TARGET = 'fare_amount'
MODEL_FILE, META_FILE, METRICS_FILE = 'uber_fare_model.pkl', 'uber_fare_metadata.json', 'metrics.json'
# Only the columns the cleaning rules and the model need; identifiers are never loaded.
USE_COLUMNS = RAW_FEATURES + [TARGET, 'distance']


def load_config(path=BASE_DIR / 'config.json'):
    with open(path) as f:
        return json.load(f)


def load_data(path, url):
    """Read the trips, downloading the file first if it is not there yet."""
    path = Path(path)
    if not path.exists():
        import gdown
        print(f'{path} not found - downloading it')
        gdown.download(url, str(path), fuzzy=True, quiet=True)
    raw = pd.read_csv(path, usecols=USE_COLUMNS)
    raw['pickup_datetime'] = pd.to_datetime(raw['pickup_datetime'], utc=True, errors='coerce').dt.tz_localize(None)
    return raw


def clean_trips(raw, rules):
    """The validity rules of notebook section 2.3, applied in the same order. Returns (clean, removed per rule)."""
    box = rules['nyc_box']

    def in_box(lat, lon):
        return lat.between(box['lat_min'], box['lat_max']) & lon.between(box['lon_min'], box['lon_max'])

    pickup_lat, pickup_lon = np.degrees(raw['pickup_latitude']), np.degrees(raw['pickup_longitude'])
    dropoff_lat, dropoff_lon = np.degrees(raw['dropoff_latitude']), np.degrees(raw['dropoff_longitude'])
    masks = {
        'missing value': raw.isna().any(axis=1),
        'fare below minimum': raw[TARGET] < rules['min_fare'],
        'passenger count out of range': ~raw['passenger_count'].between(rules['passenger_min'], rules['passenger_max']),
        'pickup outside NYC area': ~in_box(pickup_lat, pickup_lon),
        'dropoff outside NYC area': ~in_box(dropoff_lat, dropoff_lon),
        'distance = 0': raw['distance'] == 0,
        'fare > $100 for < 1 km': (raw[TARGET] > 100) & (raw['distance'] < 1),
    }
    keep = pd.Series(True, index=raw.index)
    removed = {}
    for name, mask in masks.items():
        removed[name] = int((mask & keep).sum())
        keep &= ~mask
    return raw[keep], removed


def build_pipeline(model_params, random_state):
    """features -> imputer / one-hot encoder -> gradient boosting, as in notebook section 6."""
    preprocess = ColumnTransformer(
        [('num', Pipeline([('impute', SimpleImputer(strategy='median'))]), list(NUM_FEATURES)),
         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CAT_FEATURES))],
        remainder='drop', verbose_feature_names_out=False)
    return Pipeline([('features', FunctionTransformer(add_trip_features)),
                     ('prep', preprocess),
                     ('model', HistGradientBoostingRegressor(random_state=random_state, **model_params))])


def regression_metrics(y_true, y_pred):
    return {'MAE': float(mean_absolute_error(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'R2': float(r2_score(y_true, y_pred))}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=BASE_DIR, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(data_path, output_dir, max_mae=None, config=None):
    """Train, evaluate and gate. Returns (passed, metrics). Model files are written only when the gate passes."""
    config = config or load_config()
    max_mae = config['quality_gate']['max_test_mae'] if max_mae is None else max_mae
    seed, rules = config['random_state'], config['cleaning']

    raw = load_data(data_path, config['data_url'])
    clean, removed = clean_trips(raw, rules)
    print(f'Loaded {len(raw):,} trips, kept {len(clean):,} after the validity rules.')

    X, y = clean[RAW_FEATURES], clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=seed)

    start = time.perf_counter()
    pipeline = build_pipeline(config['model_params'], seed).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    test = regression_metrics(y_test, pipeline.predict(X_test))
    baseline = DummyRegressor(strategy='median').fit(X_train, y_train)
    passed = test['MAE'] <= max_mae
    metrics = {
        'model_name': config['model_name'],
        'test': test,
        'train_MAE': float(mean_absolute_error(y_train, pipeline.predict(X_train))),
        'baseline_median_MAE': float(mean_absolute_error(y_test, baseline.predict(X_test))),
        'quality_gate': {'max_test_mae': max_mae, 'passed': bool(passed)},
        'rows': {'loaded': len(raw), 'clean': len(clean), 'train': len(X_train), 'test': len(X_test)},
        'rows_removed': removed,
        'fit_seconds': round(fit_seconds, 1),
        'data_sha256': file_sha256(data_path),
        'git_commit': git_commit(),
        'sklearn_version': sklearn.__version__,
    }
    print(f"Test MAE ${test['MAE']:.3f}  RMSE ${test['RMSE']:.3f}  R2 {test['R2']:.3f}  "
          f"(median baseline MAE ${metrics['baseline_median_MAE']:.3f}, fit {fit_seconds:.0f}s)")

    if not passed:
        print(f"QUALITY GATE FAILED: test MAE ${test['MAE']:.3f} is above the limit of ${max_mae:.2f}. Nothing was saved.")
        return False, metrics

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / MODEL_FILE, compress=3)
    box = rules['nyc_box']
    metadata = {
        'model_name': config['model_name'],
        'trained_on_rows': len(X_train),
        'sklearn_version': sklearn.__version__,
        'raw_features': RAW_FEATURES,
        'coordinate_unit': 'radians',
        'limits': {**box, 'passenger_min': rules['passenger_min'], 'passenger_max': rules['passenger_max']},
        'categories': {col: sorted(str(level) for level in X_train[col].unique()) for col in CAT_FEATURES},
        'training_period': [str(clean['pickup_datetime'].min()), str(clean['pickup_datetime'].max())],
        'test_metrics': test,
    }
    with open(output_dir / META_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Quality gate passed (limit ${max_mae:.2f}). Saved {MODEL_FILE}, {META_FILE} and {METRICS_FILE} to {output_dir}.")
    return True, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=str(BASE_DIR / 'data.csv'), help='CSV of trips; downloaded if missing')
    parser.add_argument('--output-dir', default=str(BASE_DIR), help='where the model files are written')
    parser.add_argument('--max-mae', type=float, default=None, help='quality gate: highest test MAE allowed (default: config.json)')
    args = parser.parse_args(argv)
    passed, _ = train(args.data, args.output_dir, args.max_mae)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
