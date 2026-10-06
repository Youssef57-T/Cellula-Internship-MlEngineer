import numpy as np
import pandas as pd
import pytest

from conftest import GRAND_CENTRAL, JFK, TIMES_SQUARE
from trip_features import CAT_FEATURES, NUM_FEATURES, RAW_FEATURES, add_trip_features, exp_fare, haversine_km, log_fare


def raw_trip(pickup, dropoff, when='2014-06-12 18:30:00'):
    row = {'pickup_datetime': when,
           'pickup_latitude': np.radians(pickup[0]), 'pickup_longitude': np.radians(pickup[1]),
           'dropoff_latitude': np.radians(dropoff[0]), 'dropoff_longitude': np.radians(dropoff[1]),
           'passenger_count': 1, 'Car Condition': 'Good', 'Weather': 'sunny', 'Traffic Condition': 'Flow Traffic'}
    return pd.DataFrame([row])[RAW_FEATURES]


def test_haversine_known_distances():
    to_radians = lambda point: (np.radians(point[0]), np.radians(point[1]))
    assert haversine_km(*to_radians(TIMES_SQUARE), *to_radians(JFK)) == pytest.approx(21.77, abs=0.05)
    assert haversine_km(*to_radians(TIMES_SQUARE), *to_radians(TIMES_SQUARE)) == 0


def test_feature_columns_match_the_model_inputs():
    features = add_trip_features(raw_trip(TIMES_SQUARE, GRAND_CENTRAL))
    assert list(features.columns) == NUM_FEATURES + CAT_FEATURES
    assert not features[NUM_FEATURES].isna().any().any()


def test_airport_flags():
    to_airport = add_trip_features(raw_trip(TIMES_SQUARE, JFK)).iloc[0]
    assert (to_airport['airport_pickup'], to_airport['airport_dropoff']) == (0, 1)
    from_airport = add_trip_features(raw_trip(JFK, TIMES_SQUARE)).iloc[0]
    assert (from_airport['airport_pickup'], from_airport['airport_dropoff']) == (1, 0)
    assert to_airport['dropoff_jfk_km'] == pytest.approx(0, abs=0.01)


@pytest.mark.parametrize('when, night, rush, weekend', [
    ('2014-06-12 18:30:00', 0, 1, 0),   # Thursday evening
    ('2014-06-12 23:00:00', 1, 0, 0),   # Thursday night
    ('2014-06-14 18:30:00', 0, 0, 1),   # Saturday evening: no weekday rush
    ('2014-06-12 05:59:00', 1, 0, 0),
    ('2014-06-12 06:00:00', 0, 0, 0),
])
def test_time_flags(when, night, rush, weekend):
    features = add_trip_features(raw_trip(TIMES_SQUARE, GRAND_CENTRAL, when)).iloc[0]
    assert (features['is_night'], features['is_rush_hour'], features['is_weekend']) == (night, rush, weekend)


def test_log_fare_round_trip():
    fares = np.array([2.5, 8.5, 52.0])
    assert exp_fare(log_fare(fares)) == pytest.approx(fares)
