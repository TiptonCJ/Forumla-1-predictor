import os
import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import fastf1.plotting as f1plot

# Ensure the cache directory exists
os.makedirs('fastf1_cache', exist_ok=True)

# f1_prediction_pipeline.py
fastf1.Cache.enable_cache("fastf1_cache")

# f1_prediction_pipeline.py

# === Config ===
SEASONS_AND_ROUNDS = [
    (2023, list(range(1, 23))),  # 2023: 22 rounds
    (2024, list(range(1, 23))),  # 2024: 22 rounds
    (2025, list(range(1, 10)))   # 2025: 10 rounds so far
]
PREDICT_YEAR = 2025
PREDICT_ROUND = 11
PREDICTION_MODE = 'both'  # options: 'both', 'race_only'

# === Helper Functions ===
def get_session_results(year, round_num):
    try:
        qualy = fastf1.get_session(year, round_num, 'Q')
        race = fastf1.get_session(year, round_num, 'R')
        qualy.load()
        race.load()
        print(qualy.results.columns)
        print(qualy.results.head())

        qualy_df = qualy.results[['DriverId', 'Position', 'TeamName']].rename(
            columns={'Position': 'QualyPos', 'TeamName': 'Constructor'})
        race_df = race.results[['DriverId', 'Position']].rename(columns={'Position': 'RacePos'})

        merged = pd.merge(qualy_df, race_df, on='DriverId')
        merged['Round'] = round_num
        return merged
    except Exception as e:
        print(f"Skipping round {round_num} due to error: {e}")
        return pd.DataFrame()

# === Aggregate Data ===
all_data = pd.DataFrame()

for season, rounds in SEASONS_AND_ROUNDS:
    for rnd in rounds:
        race_data = get_session_results(season, rnd)
        all_data = pd.concat([all_data, race_data], ignore_index=True)

# === Filter out data from the predicted race and any after it ===
#all_data = all_data[all_data['Year'] <= PREDICT_YEAR]
all_data = all_data[all_data['Round'] < PREDICT_ROUND]

# === Feature Engineering ===
# Average prior race/qualy position for each driver
all_data['ConstructorCode'] = all_data['Constructor'].astype('category').cat.codes

# Lag features: average qualy/race pos prior to each round
def add_avg_stats(df):
    df = df.sort_values(by=['DriverId', 'Round'])
    df['AvgQualyPos'] = df.groupby('DriverId')['QualyPos'].transform(lambda x: x.shift().expanding().mean())
    df['AvgRacePos'] = df.groupby('DriverId')['RacePos'].transform(lambda x: x.shift().expanding().mean())
    return df.dropna()

all_data = add_avg_stats(all_data)

# === Predict Qualifying ===
Xq = all_data[['AvgQualyPos', 'AvgRacePos', 'ConstructorCode']]
yq = all_data['QualyPos']

Xq_train, Xq_test, yq_train, yq_test = train_test_split(Xq, yq, test_size=0.2, random_state=42)

qualy_model = RandomForestRegressor(n_estimators=100, random_state=42)
qualy_model.fit(Xq_train, yq_train)

qualy_preds = qualy_model.predict(Xq_test)
print("Qualifying MAE:", mean_absolute_error(yq_test, qualy_preds))

# === Predict Race Using Predicted Qualy ===
all_data.loc[Xq_test.index, 'PredQualy'] = qualy_preds

Xr = all_data.loc[Xq_test.index][['PredQualy', 'AvgQualyPos', 'AvgRacePos', 'ConstructorCode']]
yr = all_data.loc[Xq_test.index]['RacePos']

race_model = RandomForestRegressor(n_estimators=100, random_state=42)
race_model.fit(Xr, yr)

race_preds = race_model.predict(Xr)
print("Race MAE:", mean_absolute_error(yr, race_preds))

# === Predict for 2025 Round 11 (Austria) ===
# Use only active drivers and teams from the most recent race
try:
    recent_race = fastf1.get_session(PREDICT_YEAR, PREDICT_ROUND - 1, 'R')
    recent_race.load()
    active_results = recent_race.results[['DriverId', 'TeamName']].drop_duplicates()
    pred_input = []
    for _, row in active_results.iterrows():
        driver = row['DriverId']
        constructor = row['TeamName']
        driver_hist = all_data[(all_data['DriverId'] == driver) & (all_data['Constructor'] == constructor)]
        if not driver_hist.empty:
            avg_qualy = driver_hist['QualyPos'].mean()
            avg_race = driver_hist['RacePos'].mean()
            constructor_code = driver_hist['ConstructorCode'].iloc[-1]
            pred_input.append({
                'AvgQualyPos': avg_qualy,
                'AvgRacePos': avg_race,
                'ConstructorCode': constructor_code,
                'DriverId': driver,
                'Constructor': constructor
            })
except Exception as e:
    print(f"Could not load most recent race for active driver/team filtering: {e}")
    pred_input = []
    fallback = all_data[all_data['Round'] == (PREDICT_ROUND - 1)]
    for _, row in fallback.iterrows():
        driver = row['DriverId']
        constructor = row['Constructor']
        driver_hist = all_data[(all_data['DriverId'] == driver) & (all_data['Constructor'] == constructor)]
        if not driver_hist.empty:
            avg_qualy = driver_hist['QualyPos'].mean()
            avg_race = driver_hist['RacePos'].mean()
            constructor_code = driver_hist['ConstructorCode'].iloc[-1]
            pred_input.append({
                'AvgQualyPos': avg_qualy,
                'AvgRacePos': avg_race,
                'ConstructorCode': constructor_code,
                'DriverId': driver,
                'Constructor': constructor
            })
pred_df = pd.DataFrame(pred_input)

if PREDICTION_MODE == 'race_only':
    # Try to fetch actual qualifying results for round 11
    try:
        qualy = fastf1.get_session(PREDICT_YEAR, PREDICT_ROUND, 'Q')
        qualy.load()
        qualy_results = qualy.results[['DriverId', 'Position']].rename(columns={'Position': 'QualyPos'})
        pred_df = pred_df.merge(qualy_results, on='DriverId', how='left')
        if pred_df['QualyPos'].isnull().any():
            print("Warning: Some drivers missing qualifying results for round 11. Using predicted qualy for those drivers.")
            pred_df['QualyPos'] = pred_df['QualyPos'].fillna(qualy_model.predict(pred_df[['AvgQualyPos', 'AvgRacePos', 'ConstructorCode']]))
        pred_df['PredQualy'] = pred_df['QualyPos']
    except Exception as e:
        print(f"Could not fetch qualifying results for round 11: {e}. Predicting both qualy and race.")
        pred_df['PredQualy'] = qualy_model.predict(pred_df[['AvgQualyPos', 'AvgRacePos', 'ConstructorCode']])
else:
    # Predict both qualy and race
    pred_df['PredQualy'] = qualy_model.predict(pred_df[['AvgQualyPos', 'AvgRacePos', 'ConstructorCode']])

race_pred = race_model.predict(pred_df[['PredQualy', 'AvgQualyPos', 'AvgRacePos', 'ConstructorCode']])
pred_df['PredRacePos'] = race_pred
# Assign unique predicted qualifying positions (1, 2, 3, ...) based on sorted PredQualy
pred_df = pred_df.sort_values('PredQualy').reset_index(drop=True)
pred_df['PredQualy'] = range(1, len(pred_df) + 1)
# Assign unique race positions (1, 2, 3, ...) based on sorted PredRacePos
pred_df = pred_df.sort_values('PredRacePos').reset_index(drop=True)
pred_df['PredRacePos'] = range(1, len(pred_df) + 1)
print("\nPredicted Results for " + PREDICT_YEAR + " Round " + str(PREDICT_ROUND) + ":\n")
print(pred_df[['DriverId', 'Constructor', 'PredQualy', 'PredRacePos']])

