import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Paths
base_path = r'C:\Users\Lydiah\PyCharmProjects\PythonProject\PythonProject\March ML Mania 2025\data\march-machine-learning-mania-2025\\'

# Load team strength features
team_strength = pd.read_csv(base_path + 'team_strength.csv')

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

# ✅ **Create both win (1) and loss (0) cases**
tourney_results_win = tourney_results.copy()
tourney_results_win['Target'] = 1  # Winner is 1

tourney_results_loss = tourney_results.copy()
tourney_results_loss = tourney_results_loss.rename(columns={
    'WTeamID': 'LTeamID', 'LTeamID': 'WTeamID',  # Swap teams
    'WTeamWinRatio': 'LTeamWinRatio', 'LTeamWinRatio': 'WTeamWinRatio'
})
tourney_results_loss['Target'] = 0  # Loser is 0

# Combine both win and loss cases
final_tourney_data = pd.concat([tourney_results_win, tourney_results_loss])

# Features and Target
X = final_tourney_data[['WTeamWinRatio', 'LTeamWinRatio']]
y = final_tourney_data['Target']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict_proba(X_val)
loss = log_loss(y_val, y_pred)

print(f"🚀 Baseline Log Loss: {loss:.4f}")
