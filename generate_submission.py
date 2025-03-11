import pandas as pd
import xgboost as xgb

base_path = r'C:\Users\Lydiah\PyCharmProjects\PythonProject\PythonProject\March ML Mania 2025\data\march-machine-learning-mania-2025\\'

# Load trained model
model = xgb.XGBClassifier()
model.load_model(base_path + 'trained_model.json')

# Load tournament matchups
sample_submission = pd.read_csv(base_path + 'SampleSubmissionStage1.csv')

# Extract Season, Team1, Team2 from the ID column
sample_submission[['Season', 'Team1', 'Team2']] = sample_submission['ID'].str.split('_', expand=True).astype(int)

# Load engineered features
team_strength = pd.read_csv(base_path + 'team_strength.csv')
seeds_df = pd.read_csv(base_path + 'team_seeds.csv')
recent_performance = pd.read_csv(base_path + 'team_recent_performance.csv')

# Merge features for Team1
sample_submission = sample_submission.merge(team_strength, left_on='Team1', right_on='TeamID', how='left')
sample_submission = sample_submission.rename(columns={'WinRatio': 'Team1WinRatio'}).drop(['TeamID', 'Wins', 'Losses'], axis=1)

# Merge features for Team2
sample_submission = sample_submission.merge(team_strength, left_on='Team2', right_on='TeamID', how='left')
sample_submission = sample_submission.rename(columns={'WinRatio': 'Team2WinRatio'}).drop(['TeamID', 'Wins', 'Losses'], axis=1)

# Merge tournament seed info
sample_submission = sample_submission.merge(seeds_df, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
sample_submission = sample_submission.rename(columns={'SeedNum': 'Team1Seed'}).drop(['TeamID'], axis=1)

sample_submission = sample_submission.merge(seeds_df, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')
sample_submission = sample_submission.rename(columns={'SeedNum': 'Team2Seed'}).drop(['TeamID'], axis=1)

# Compute Seed Difference
sample_submission['SeedDiff'] = sample_submission['Team2Seed'] - sample_submission['Team1Seed']

# Merge recent performance for Team1
sample_submission = sample_submission.merge(recent_performance, left_on=['Season', 'Team1'], right_on=['Season', 'WTeamID'], how='left')
sample_submission = sample_submission.rename(columns={'WinRatioLast10': 'Team1RecentWinPct'}).drop(['WTeamID'], axis=1)

# Merge recent performance for Team2
sample_submission = sample_submission.merge(recent_performance, left_on=['Season', 'Team2'], right_on=['Season', 'WTeamID'], how='left')
sample_submission = sample_submission.rename(columns={'WinRatioLast10': 'Team2RecentWinPct'}).drop(['WTeamID'], axis=1)

# **Fix Feature Name Mismatch**
sample_submission = sample_submission.rename(columns={
    'Team1WinRatio': 'WTeamWinRatio',
    'Team2WinRatio': 'LTeamWinRatio',
    'Team1RecentWinPct': 'WTeamRecentWinPct',
    'Team2RecentWinPct': 'LTeamRecentWinPct'
})

# Prepare feature set
X_test = sample_submission[['WTeamWinRatio', 'LTeamWinRatio', 'SeedDiff', 'WTeamRecentWinPct', 'LTeamRecentWinPct']]

# Predict probabilities
sample_submission['Pred'] = model.predict_proba(X_test)[:, 1]

# Save submission file
sample_submission[['ID', 'Pred']].to_csv(base_path + 'submission.csv', index=False)

print("🚀 Kaggle Submission File Generated: submission.csv ✅")
