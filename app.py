"""Flask app that serves taxi-fare predictions from the pipeline saved by Task3.ipynb.

Run with:  python app.py   then open http://127.0.0.1:5000
"""
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

# The saved pipeline refers to the functions in trip_features.py, so the module must be importable.
from trip_features import add_trip_features, haversine_km

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'uber_fare_model.pkl'
META_PATH = BASE_DIR / 'uber_fare_metadata.json'

EARLIEST_DATE = datetime(2009, 1, 1)
DATETIME_FORMATS = ('%Y-%m-%dT%H:%M', '%Y-%m-%dT%H:%M:%S')
COORD_FIELDS = {
    'pickup_lat': ('Pickup latitude', 'lat'),
    'pickup_lon': ('Pickup longitude', 'lon'),
    'dropoff_lat': ('Dropoff latitude', 'lat'),
    'dropoff_lon': ('Dropoff longitude', 'lon'),
}
# Any point on Earth is accepted; the model itself was trained on New York trips only.
COORD_MAX = {'lat': 90, 'lon': 180}
CATEGORY_FIELDS = {
    'car_condition': 'Car Condition',
    'weather': 'Weather',
    'traffic_condition': 'Traffic Condition',
}
PREFERRED_DEFAULTS = {'car_condition': 'Good', 'weather': 'sunny', 'traffic_condition': 'Flow Traffic'}
LANDMARKS = {
    'Times Square': (40.7580, -73.9855),
    'Grand Central': (40.7527, -73.9772),
    'Wall Street': (40.7060, -74.0088),
    'Central Park (72nd St)': (40.7760, -73.9760),
    'JFK Airport': (40.6413, -73.7781),
    'LaGuardia Airport': (40.7769, -73.8740),
}

# Loaded once, when the app starts - never per request.
if not MODEL_PATH.exists() or not META_PATH.exists():
    raise SystemExit(f"Missing {MODEL_PATH.name} or {META_PATH.name}. Run section 12 of Task3.ipynb first.")
pipeline = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
LIMITS = meta['limits']
TRAINING_END = datetime.fromisoformat(meta['training_period'][1])

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# The models the deployed one was compared with. Their fares are shown next to the main one.
other_models = []
for entry in meta.get('other_models', []):
    if (BASE_DIR / entry['file']).exists():
        other_models.append({'name': entry['name'], 'mae': entry['test_metrics']['MAE'], 'r2': entry['test_metrics']['R2'],
                             'pipeline': joblib.load(BASE_DIR / entry['file'])})
    else:
        app.logger.warning('%s is missing, so "%s" is not shown. Run train.py to create it.', entry['file'], entry['name'])


def default_form():
    """A complete, valid trip, so the form is never empty."""
    form = {
        'pickup_datetime': '2014-06-12T18:30',
        'pickup_lat': str(LANDMARKS['Times Square'][0]), 'pickup_lon': str(LANDMARKS['Times Square'][1]),
        'dropoff_lat': str(LANDMARKS['Grand Central'][0]), 'dropoff_lon': str(LANDMARKS['Grand Central'][1]),
        'passenger_count': '1',
    }
    for field, column in CATEGORY_FIELDS.items():
        levels = meta['categories'][column]
        form[field] = PREFERRED_DEFAULTS[field] if PREFERRED_DEFAULTS[field] in levels else levels[0]
    return form


def parse_form(form):
    """Check the submitted strings. Locations may be anywhere; the other fields follow the Task 2 cleaning rules.

    Returns (values, errors): typed values for the fields that are valid, and one message per invalid field.
    """
    values, errors = {}, {}

    raw = form.get('pickup_datetime', '').strip()
    if not raw:
        errors['pickup_datetime'] = 'Pickup date and time is required.'
    else:
        for fmt in DATETIME_FORMATS:
            try:
                values['pickup_datetime'] = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        if 'pickup_datetime' not in values:
            errors['pickup_datetime'] = 'Enter a valid pickup date and time.'
        elif values['pickup_datetime'] < EARLIEST_DATE:
            del values['pickup_datetime']
            errors['pickup_datetime'] = 'Pickup date must be 1 January 2009 or later.'

    for field, (label, axis) in COORD_FIELDS.items():
        high = COORD_MAX[axis]
        raw = form.get(field, '').strip()
        if not raw:
            errors[field] = f'{label} is required.'
            continue
        try:
            number = float(raw)
        except ValueError:
            number = math.nan
        if not math.isfinite(number):
            errors[field] = f'{label} must be a number.'
        elif not -high <= number <= high:
            errors[field] = f'{label} must be between -{high} and {high}.'
        else:
            values[field] = number

    if all(field in values for field in COORD_FIELDS):
        km = haversine_km(*(math.radians(values[f]) for f in ('pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon')))
        if km == 0:
            errors['dropoff_lat'] = 'Pickup and dropoff are the same point. Enter two different locations.'

    low, high = LIMITS['passenger_min'], LIMITS['passenger_max']
    raw = form.get('passenger_count', '').strip()
    if not raw:
        errors['passenger_count'] = 'Number of passengers is required.'
    elif not re.fullmatch(r'[0-9]{1,9}', raw) or not low <= int(raw) <= high:
        errors['passenger_count'] = f'Passengers must be a whole number from {low} to {high}.'
    else:
        values['passenger_count'] = int(raw)

    for field, column in CATEGORY_FIELDS.items():
        raw = form.get(field, '').strip()
        if raw in meta['categories'][column]:
            values[field] = raw
        else:
            errors[field] = f'Choose a value for {column.lower()} from the list.'

    return values, errors


def to_model_frame(values):
    """One raw trip in the layout of the training data: coordinates in radians, columns in training order."""
    row = {
        'pickup_datetime': values['pickup_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
        'pickup_longitude': math.radians(values['pickup_lon']),
        'pickup_latitude': math.radians(values['pickup_lat']),
        'dropoff_longitude': math.radians(values['dropoff_lon']),
        'dropoff_latitude': math.radians(values['dropoff_lat']),
        'passenger_count': values['passenger_count'],
    }
    for field, column in CATEGORY_FIELDS.items():
        row[column] = values[field]
    return pd.DataFrame([row])[meta['raw_features']]


def trip_summary(frame):
    """Figures a person can check by hand, computed with the same feature code the model uses."""
    features = add_trip_features(frame).iloc[0]
    return {
        'trip_km': float(features['trip_km']),
        'airport_trip': bool(features['airport_pickup'] or features['airport_dropoff']),
    }


def model_rows(frame=None):
    """Every loaded model with its test-set MAE and R2, most accurate first. With a trip, also the fare each one predicts."""
    deployed = {'name': meta['model_name'], 'mae': meta['test_metrics']['MAE'], 'r2': meta['test_metrics']['R2'],
                'pipeline': pipeline}
    rows = []
    for model in sorted([deployed] + other_models, key=lambda model: model['mae']):
        rows.append({'name': model['name'], 'mae': model['mae'], 'r2': model['r2'], 'deployed': model is deployed,
                     'fare': None if frame is None else float(model['pipeline'].predict(frame)[0])})
    # The real fare of a submitted trip is unknown, so the best model is the one with the lowest error on the test set.
    for position, row in enumerate(rows):
        row['best'] = position == 0
    return rows


def render_page(form, errors=None, result=None, status=200):
    return render_template('index.html', form=form, errors=errors or {}, result=result, meta=meta, limits=LIMITS,
                           models=result['models'] if result else model_rows(),
                           landmarks=LANDMARKS, category_fields=CATEGORY_FIELDS), status


@app.get('/')
def index():
    return render_page(default_form())


@app.post('/predict')
def predict():
    form = {**default_form(), **request.form.to_dict()}
    values, errors = parse_form(request.form)
    if errors:
        return render_page(form, errors=errors, status=400)

    frame = to_model_frame(values)
    models = model_rows(frame)
    fare = next(model['fare'] for model in models if model['deployed'])
    result = {
        'fare': fare,
        'models': models,
        'typical_error': meta['test_metrics']['MAE'],
        'after_training_period': values['pickup_datetime'] > TRAINING_END,
        'pickup_time': values['pickup_datetime'].strftime('%a %d %b %Y, %H:%M'),
        'passengers': values['passenger_count'],
        **trip_summary(frame),
    }
    app.logger.info('Predicted %.2f for %s', fare, frame.iloc[0].to_dict())
    return render_page(form, result=result)


@app.get('/health')
def health():
    return jsonify(status='ok', model=meta['model_name'], sklearn_version=meta['sklearn_version'],
                   test_mae=meta['test_metrics']['MAE'], other_models=[other['name'] for other in other_models])


@app.errorhandler(Exception)
def unexpected_error(error):
    if isinstance(error, HTTPException):
        return error
    app.logger.exception('Unexpected error while handling %s', request.path)
    form = {**default_form(), **request.form.to_dict()}
    return render_page(form, errors={'form': 'Something went wrong while predicting this trip. Please check the values and try again.'},
                       status=500)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
