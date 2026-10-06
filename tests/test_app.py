import re

import pytest

import app as fare_app
from conftest import JFK, TIMES_SQUARE


@pytest.fixture
def client():
    return fare_app.app.test_client()


def shown_fare(response):
    match = re.search(r'class="fare">\$([\d.]+)', response.get_data(as_text=True))
    return float(match.group(1)) if match else None


def messages(response):
    return re.findall(r'<li>(.*?)</li>', response.get_data(as_text=True))


def test_form_page_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Predict fare' in response.data


def test_health(client):
    body = client.get('/health').get_json()
    assert body['status'] == 'ok'
    assert body['test_mae'] > 0


def test_valid_trip_returns_the_pipeline_prediction(client, trip_form):
    form = trip_form()
    response = client.post('/predict', data=form)
    assert response.status_code == 200

    # The page must show exactly what the saved pipeline predicts for the same raw trip.
    values, errors = fare_app.parse_form({name: str(value) for name, value in form.items()})
    assert not errors
    expected = float(fare_app.pipeline.predict(fare_app.to_model_frame(values))[0])
    assert shown_fare(response) == pytest.approx(expected, abs=0.006)


def test_airport_trip_is_recognised(client, trip_form):
    response = client.post('/predict', data=trip_form(dropoff=JFK))
    assert response.status_code == 200
    assert '<dd>Yes</dd>' in response.get_data(as_text=True)


def test_date_after_training_period_shows_a_note(client, trip_form):
    response = client.post('/predict', data=trip_form(pickup_datetime='2026-10-06T18:30'))
    assert response.status_code == 200
    assert 'after the training data ends' in response.get_data(as_text=True)


@pytest.mark.parametrize('overrides, expected', [
    ({'dropoff_lat': TIMES_SQUARE[0], 'dropoff_lon': TIMES_SQUARE[1]}, 'same point'),
    ({'passenger_count': '0'}, 'whole number from 1 to 6'),
    ({'passenger_count': '1.5'}, 'whole number from 1 to 6'),
    ({'passenger_count': ''}, 'passengers is required'),
    ({'pickup_datetime': ''}, 'date and time is required'),
    ({'pickup_datetime': 'tomorrow'}, 'valid pickup date'),
    ({'pickup_datetime': '1999-01-01T10:00'}, '2009 or later'),
    ({'dropoff_lat': '51.5', 'dropoff_lon': '-0.12'}, 'New York area'),
    ({'pickup_lat': 'abc'}, 'must be a number'),
    ({'pickup_lat': 'nan'}, 'must be a number'),
    ({'dropoff_lon': ''}, 'longitude is required'),
    ({'weather': 'snow'}, 'weather from the list'),
])
def test_invalid_input_gives_a_message_and_no_fare(client, trip_form, overrides, expected):
    response = client.post('/predict', data=trip_form(**overrides))
    assert response.status_code == 400
    assert shown_fare(response) is None
    assert any(expected in message for message in messages(response))


def test_empty_submission_lists_every_field(client):
    response = client.post('/predict', data={})
    assert response.status_code == 400
    assert len(messages(response)) == 9


def test_unexpected_failure_shows_a_message_not_a_traceback(client, trip_form, monkeypatch):
    class Broken:
        def predict(self, frame):
            raise RuntimeError('boom')

    monkeypatch.setattr(fare_app, 'pipeline', Broken())
    response = client.post('/predict', data=trip_form())
    assert response.status_code == 500
    assert b'Something went wrong' in response.data
    assert b'Traceback' not in response.data
