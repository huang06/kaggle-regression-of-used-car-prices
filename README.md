# Kaggle Regression of Used Car Prices

<https://www.kaggle.com/competitions/playground-series-s4e9>

## Key Features

**Feature Engineering**

- Created new features like is_luxury (for luxury brands) and milage_per_year (for usage intensity).
- Combined brand and model into one feature (brand_model) to capture relationships between the two.

**Outlier Detection** (Inspired by Top Kaggle Competitor)

- Observed that car prices have a right-skewed distribution, with a small number of extremely high-priced cars.
- used the interquartile range (IQR) to identify high-priced outliers and creates a new binary feature (price_bin) to separate these cars from the rest, allowing the model to handle them differently.

**Target Encoding**

- Applied target encoding to convert categorical variables (like brand and model) into numerical features based on their median prices.

**Model Stacking with Out-of-Fold Predictions**

- Generated out-of-fold (OOF) predictions for each base model and feeding them into a meta-learner.
- The OOF predictions allow the meta-learner to make better final predictions by leveraging the strengths of each base model.

**Final Ensemble Model**

- The final model aggregates the predictions from base models into a Ridge regression model, which serves as the meta-learner. This approach takes advantage of each model’s strengths to improve the overall prediction performance.

## Installation

- Python3 (testd with 3.10)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -U -r requirements.txt
```
