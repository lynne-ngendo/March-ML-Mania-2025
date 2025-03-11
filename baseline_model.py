import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

base_path = r'C:\Users\Lydiah\PyCharmProjects\PythonProject\PythonProject\March ML Mania 2025\data\march-machine-learning-mania-2025\\'

# Load data
team_strength = pd.read_csv(base_path + 'team_strength.csv')
seeds_df = pd.read_csv(base_path + 'team_seeds.csv')
recent_performance = pd.read_csv(base_path + 'team_recent_performance.csv')

# Load tournament results
tourney_results = pd.read_csv(base_path + 'MNCAATourneyCompactResults.csv')

# Merge team strength with winners
tourney_results = tourney_results.merge(
    team_strength, left_on='WTeamID', right_on='TeamID', how='left'
).rename(columns={'WinRatio': 'WTeamWinRatio'}).drop(['TeamID', 'Wins', 'Losses'], axis=1)

# Merge team strength with losers
tourney_results = tourney_results.merge(
    team_strength, left_on='LTeamID', right_on='TeamID', how='left'
).rename(columns={'WinRatio': 'LTeamWinRatio'}).drop(['TeamID', 'Wins', 'Losses'], axis=1)

# Merge tournament seed data
tourney_results = tourney_results.merge(seeds_df, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
tourney_results = tourney_results.rename(columns={'SeedNum': 'WTeamSeed'}).drop(['TeamID'], axis=1)

tourney_results = tourney_results.merge(seeds_df, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
tourney_results = tourney_results.rename(columns={'SeedNum': 'LTeamSeed'}).drop(['TeamID'], axis=1)

tourney_results['SeedDiff'] = tourney_results['LTeamSeed'] - tourney_results['WTeamSeed']

# Merge recent performance
tourney_results = tourney_results.merge(recent_performance, left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'], how='left')
tourney_results = tourney_results.rename(columns={'WinRatioLast10': 'WTeamRecentWinPct'})

tourney_results = tourney_results.merge(recent_performance, left_on=['Season', 'LTeamID'], right_on=['Season', 'WTeamID'], how='left')
tourney_results = tourney_results.rename(columns={'WinRatioLast10': 'LTeamRecentWinPct'})

# **Create both win (1) and loss (0) cases**
tourney_results_win = tourney_results.copy()
tourney_results_win['Target'] = 1

tourney_results_loss = tourney_results.copy()
tourney_results_loss = tourney_results_loss.rename(columns={
    'WTeamID': 'LTeamID', 'LTeamID': 'WTeamID',
    'WTeamWinRatio': 'LTeamWinRatio', 'LTeamWinRatio': 'WTeamWinRatio'
})
tourney_results_loss['Target'] = 0

final_tourney_data = pd.concat([tourney_results_win, tourney_results_loss])

# Features & Target
X = final_tourney_data[['WTeamWinRatio', 'LTeamWinRatio', 'SeedDiff', 'WTeamRecentWinPct', 'LTeamRecentWinPct']]
y = final_tourney_data['Target']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model with optimized parameters
model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict_proba(X_val)
loss = log_loss(y_val, y_pred)

print(f"🔥 Improved Log Loss: {loss:.4f}")
