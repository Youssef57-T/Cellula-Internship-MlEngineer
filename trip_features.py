import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
AIRPORT_RADIUS_KM = 2.0
MAX_FARE = 500.0
AIRPORTS = {'JFK': (40.6413, -73.7781), 'LGA': (40.7769, -73.8740), 'EWR': (40.6895, -74.1745)}

RAW_FEATURES = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                'passenger_count', 'Car Condition', 'Weather', 'Traffic Condition']
CAT_FEATURES = ['Car Condition', 'Weather', 'Traffic Condition']
NUM_FEATURES = ['trip_km', 'log_trip_km', 'manhattan_km', 'bearing_sin', 'bearing_cos',
                'pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon',
                'pickup_jfk_km', 'dropoff_jfk_km', 'pickup_lga_km', 'dropoff_lga_km', 'pickup_ewr_km', 'dropoff_ewr_km',
                'airport_pickup', 'airport_dropoff',
                'hour', 'hour_sin', 'hour_cos', 'weekday', 'is_weekend', 'is_night', 'is_rush_hour',
                'month', 'year', 'time_index', 'passenger_count']


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. All inputs in radians."""
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_trip_features(X):
    """Raw trip columns (coordinates in radians) -> model features. Stateless: nothing is learned from the data."""
    plat, plon = X['pickup_latitude'].astype(float), X['pickup_longitude'].astype(float)
    dlat, dlon = X['dropoff_latitude'].astype(float), X['dropoff_longitude'].astype(float)
    out = pd.DataFrame(index=X.index)

    trip_km = haversine_km(plat, plon, dlat, dlon)
    out['trip_km'] = trip_km
    out['log_trip_km'] = np.log1p(trip_km)
    out['manhattan_km'] = haversine_km(plat, plon, dlat, plon) + haversine_km(dlat, plon, dlat, dlon)
    bearing = np.arctan2(np.sin(dlon - plon) * np.cos(dlat),
                         np.cos(plat) * np.sin(dlat) - np.sin(plat) * np.cos(dlat) * np.cos(dlon - plon))
    out['bearing_sin'] = np.sin(bearing)
    out['bearing_cos'] = np.cos(bearing)

    out['pickup_lat'], out['pickup_lon'] = np.degrees(plat), np.degrees(plon)
    out['dropoff_lat'], out['dropoff_lon'] = np.degrees(dlat), np.degrees(dlon)

    for name, (lat, lon) in AIRPORTS.items():
        lat, lon = np.radians(lat), np.radians(lon)
        out[f'pickup_{name.lower()}_km'] = haversine_km(plat, plon, lat, lon)
        out[f'dropoff_{name.lower()}_km'] = haversine_km(dlat, dlon, lat, lon)
    pickup_cols = [f'pickup_{name.lower()}_km' for name in AIRPORTS]
    dropoff_cols = [f'dropoff_{name.lower()}_km' for name in AIRPORTS]
    out['airport_pickup'] = (out[pickup_cols].min(axis=1) <= AIRPORT_RADIUS_KM).astype(int)
    out['airport_dropoff'] = (out[dropoff_cols].min(axis=1) <= AIRPORT_RADIUS_KM).astype(int)

    dt = pd.to_datetime(X['pickup_datetime'], utc=True, errors='coerce')
    hour = dt.dt.hour + dt.dt.minute / 60
    out['hour'] = hour
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    out['weekday'] = dt.dt.weekday
    out['is_weekend'] = (dt.dt.weekday >= 5).astype(int)
    out['is_night'] = ((hour >= 20) | (hour < 6)).astype(int)
    out['is_rush_hour'] = ((dt.dt.weekday < 5) & (hour >= 16) & (hour < 20)).astype(int)
    out['month'] = dt.dt.month
    out['year'] = dt.dt.year
    out['time_index'] = dt.dt.year + (dt.dt.dayofyear - 1) / 365.25
    out['passenger_count'] = X['passenger_count'].astype(float)

    for col in CAT_FEATURES:
        if col in X.columns:
            out[col] = X[col]
    return out


def log_fare(y):
    return np.log1p(y)


def exp_fare(z):
    """Inverse of log_fare, capped at the largest fare in the data so one extreme trip cannot explode."""
    return np.expm1(np.clip(z, None, np.log1p(MAX_FARE)))
