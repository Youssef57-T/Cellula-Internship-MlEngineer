# NYC Taxi Fare Predictor — Task 3: Model Deployment

A Flask web app that predicts the fare of a New York taxi trip. It serves the model trained in `Task_2_v2.ipynb`.

## Run the app

Python 3.12 is required.

```
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000> in a browser.

The app needs these four files in the same folder as `app.py`:

| File | What it is |
|---|---|
| `uber_fare_model.pkl` | The saved pipeline: feature engineering, imputer, encoder and the model |
| `uber_fare_metadata.json` | Valid input ranges and category values, written by the notebook |
| `trip_features.py` | The feature functions the pipeline refers to |
| `templates/`, `static/` | The page and its stylesheet |

The model and metadata files are produced by `python train.py`, or by section 12 of the notebook; both give the same model. `scikit-learn` must stay at the version pinned in `requirements.txt`, because that is the version that saved the model.

## Files

| File | Task part |
|---|---|
| `Task_2_v2.ipynb` | EDA, preprocessing pipeline and all models. Section 12 holds Part A (comparison table and justification) and Part B (saving the model) |
| `uber_fare_model.pkl`, `uber_fare_metadata.json`, `trip_features.py` | Part B |
| `app.py`, `templates/index.html`, `static/style.css` | Part C |
| `screenshots/` | Part D |
| `train.py`, `config.json`, `metrics.json` | Training script, its settings and the scores of the committed model |
| `tests/`, `.github/workflows/` | Automated tests and the CI and retraining workflows |

## The model

Eight models were compared on a held-out test set of 96,465 trips. The tuned gradient boosting model is deployed because it has the best test result on every metric.

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **Gradient boosting (tuned)** | **1.538** | **3.308** | **0.878** |
| Gradient boosting | 1.621 | 3.394 | 0.872 |
| Random forest | 1.693 | 3.484 | 0.865 |
| Decision tree | 1.869 | 3.683 | 0.849 |
| Ridge, log target | 1.992 | 4.207 | 0.803 |
| Ridge, all features | 2.120 | 4.095 | 0.813 |
| Linear regression, distance only | 2.339 | 4.486 | 0.776 |
| Median (dummy) | 5.347 | 9.886 | −0.088 |

The full table, with training error, prediction speed and file size, and the written justification are in section 12.1 of the notebook.

## How a submitted trip is processed

Every form submission goes through the same steps as the training data, in the same order:

1. **Validation.** The input is checked with the cleaning rules from Task 2: a parsable date, 1–6 passengers, pickup and dropoff inside the New York area, and two different locations. The limits are read from `uber_fare_metadata.json`, so they are the ones the notebook used.
2. **Units.** Coordinates are typed in degrees and converted to radians, the unit of the dataset.
3. **The saved pipeline.** `pipeline.predict` runs `add_trip_features` (distance, airport distances, time features), the imputer and one-hot encoder, and then the model. The app contains no feature code of its own.

The model and metadata are loaded once when the app starts.

If the input is invalid, the page shows what to fix and no fare is predicted.

The main part of the page is a map of the trip (Leaflet with OpenStreetMap tiles). Marker **A** is the pickup and **B** the dropoff. To choose a location on the map, press **Set pickup** or **Set dropoff** and click the map, or drag a marker; either way the coordinate fields are updated. Two lines are drawn:

- the **dashed orange line** is the straight-line distance, which is what the model uses;
- the **blue line** is the route by road, fetched in the browser from a public OSRM routing server, with a car that drives along it.

The map is for display only. It reads and writes the form fields and has no effect on the prediction. It needs an internet connection; without one the form still works, and if the route service is unavailable only the straight line is shown.

## Test cases

Enter these through the form. Use the *Quick fill* lists to set the coordinates. Leave car condition, weather and traffic at their defaults (Good, sunny, Flow Traffic).

| # | Pickup | Dropoff | Date and time | Passengers | Expected result |
|---|---|---|---|---|---|
| 1 | Times Square | Grand Central | 12 Jun 2014, 18:30 | 1 | **$7.47**, 0.91 km, not an airport trip |
| 2 | Times Square | JFK Airport | 12 Jun 2014, 18:30 | 1 | **$61.86**, 21.77 km, airport trip |
| 3 | LaGuardia Airport | Wall Street | 9 Mar 2013, 09:00 | 2 | **$42.40**, 13.82 km, airport trip |
| 4 | Times Square | Times Square | any | 1 | No fare. Message: "Pickup and dropoff are the same point. Enter two different locations." |

Cases 1 and 2 are the two trips checked in section 11 of the notebook, and the app returns the same fares as that cell. This confirms the app applies the same pipeline as the notebook.

Case 4 passes the browser's own checks and is rejected by the server. Other invalid inputs, such as an empty field or 0 passengers, are stopped by the browser before the form is sent; the server rejects them too if they reach it.

## MLOps flow

The model can be rebuilt, checked and tested without opening the notebook.

```
config.json ──► train.py ──► quality gate ──► uber_fare_model.pkl ──► app.py
 (settings)    (clean, split,   (test MAE        uber_fare_metadata.json
                fit, evaluate)   ≤ $1.65)        metrics.json
```

| Step | Command | What it does |
|---|---|---|
| Train | `python train.py` | Downloads `data.csv` if it is missing, applies the Task 2 cleaning rules, trains the tuned gradient boosting pipeline and scores it on the 20% test set |
| Quality gate | part of `train.py` | If the test MAE is above the limit in `config.json` ($1.65), nothing is saved and the command exits with an error |
| Test | `pytest` | 38 tests: feature code, the app (valid and invalid input), the committed model files, and the training script on a 5,000-trip sample |
| CI | `.github/workflows/ci.yml` | On every push and pull request: runs the tests, trains on the sample with the gate, then starts the app and calls it |
| Retrain | `.github/workflows/train.yml` | Started by hand from the GitHub Actions tab: trains on the full dataset with the gate and attaches the model files to the run |

`train.py` reproduces the notebook's final model exactly: same cleaning rules, split, features and settings, and identical predictions. `metrics.json` records the scores, the row counts, the hash of the data file and the commit the model was trained from.

To install what the tests and the notebook need: `pip install -r requirements-dev.txt`.

## Known limits

- The model was trained on trips from 2009 to June 2015. For a later date the page says so and the estimate uses mid-2015 fare levels.
- Car condition, weather and traffic are inputs of the model but have no measurable effect on the prediction (notebook section 9.3).
- The JFK prediction is about $5–10 above the real flat fare (notebook section 11).
