"""Train the fare model from the command line and save it only if it passes the quality gate.

This reproduces the final model of Task3.ipynb (same cleaning rules, split, features and settings),
so the deployed model can be rebuilt without opening the notebook. The seven other models the notebook
compares it with are trained and saved too, so the app can show their fares next to the deployed one.

    python train.py                      # full data, gate from config.json
    python train.py --data sample.csv --output-dir out --max-mae 5
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from trip_features import CAT_FEATURES, NUM_FEATURES, RAW_FEATURES, add_trip_features, exp_fare, log_fare

BASE_DIR = Path(__file__).resolve().parent
TARGET = 'fare_amount'
MODEL_FILE, META_FILE, METRICS_FILE = 'uber_fare_model.pkl', 'uber_fare_metadata.json', 'metrics.json'
# The models that are compared with the deployed one are saved here, one file each.
OTHER_MODELS_DIR = 'models'
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


def make_model_pipeline(model, num=NUM_FEATURES, cat=CAT_FEATURES, scale=False, log_target=False):
    """features -> imputer (and scaler) / one-hot encoder -> model, as in notebook section 6."""
    num_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        num_steps.append(('scale', StandardScaler()))
    transformers = [('num', Pipeline(num_steps), list(num))]
    if cat:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(cat)))
    if log_target:
        model = TransformedTargetRegressor(regressor=model, func=log_fare, inverse_func=exp_fare, check_inverse=False)
    return Pipeline([('features', FunctionTransformer(add_trip_features)),
                     ('prep', ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)),
                     ('model', model)])


def build_pipeline(model_params, random_state):
    """The deployed model: tuned gradient boosting."""
    return make_model_pipeline(HistGradientBoostingRegressor(random_state=random_state, **model_params))


def other_pipelines(random_state):
    """The seven models the deployed one is compared with, with the settings of notebook section 7."""
    return {
        'Gradient boosting': make_model_pipeline(HistGradientBoostingRegressor(random_state=random_state)),
        'Random forest': make_model_pipeline(RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5,
                                                                   max_features=0.5, max_samples=0.2, n_jobs=-1,
                                                                   random_state=random_state)),
        'Decision tree': make_model_pipeline(DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=random_state)),
        'Ridge: log target': make_model_pipeline(Ridge(alpha=1.0), scale=True, log_target=True),
        'Ridge: all features': make_model_pipeline(Ridge(alpha=1.0), scale=True),
        'Linear: distance only': make_model_pipeline(LinearRegression(), num=['trip_km'], cat=None),
        'Median (dummy)': make_model_pipeline(DummyRegressor(strategy='median')),
    }


def model_file_name(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') + '.pkl'


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
    (output_dir / OTHER_MODELS_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / MODEL_FILE, compress=3)

    # The gate applies to the deployed model only; the others are kept for comparison, whatever their score.
    other_models = []
    for name, other in other_pipelines(seed).items():
        other.fit(X_train, y_train)
        file = f'{OTHER_MODELS_DIR}/{model_file_name(name)}'
        joblib.dump(other, output_dir / file, compress=3)
        other_models.append({'name': name, 'file': file, 'test_metrics': regression_metrics(y_test, other.predict(X_test))})
        print(f"{name:<24} test MAE ${other_models[-1]['test_metrics']['MAE']:.3f}")
    metrics['other_models'] = {entry['name']: entry['test_metrics'] for entry in other_models}

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
        'other_models': other_models,
    }
    with open(output_dir / META_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Quality gate passed (limit ${max_mae:.2f}). Saved {MODEL_FILE}, {META_FILE}, {METRICS_FILE} "
          f"and {len(other_models)} comparison models in {OTHER_MODELS_DIR}/ to {output_dir}.")
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
