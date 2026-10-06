import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_DATA = ROOT / 'tests' / 'data' / 'sample_trips.csv'

# Landmarks used across the tests: (latitude, longitude) in degrees.
TIMES_SQUARE = (40.7580, -73.9855)
GRAND_CENTRAL = (40.7527, -73.9772)
JFK = (40.6413, -73.7781)


@pytest.fixture
def trip_form():
    """Builds the form data of a valid trip; keyword arguments override single fields."""
    def build(pickup=TIMES_SQUARE, dropoff=GRAND_CENTRAL, **overrides):
        form = {'pickup_datetime': '2014-06-12T18:30',
                'pickup_lat': pickup[0], 'pickup_lon': pickup[1],
                'dropoff_lat': dropoff[0], 'dropoff_lon': dropoff[1],
                'passenger_count': '1', 'car_condition': 'Good', 'weather': 'sunny',
                'traffic_condition': 'Flow Traffic'}
        form.update(overrides)
        return form
    return build
