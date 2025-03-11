import pandas as pd

# Paths (confirm path correctness)
base_path = r'C:\Users\Lydiah\PyCharmProjects\PythonProject\PythonProject\March ML Mania 2025\data\march-machine-learning-mania-2025\\'

# Load regular season results
regular_season = pd.read_csv(base_path + 'MRegularSeasonCompactResults.csv')

# Compute Team Win Counts (Simple strength metric)
team_win_counts = regular_season.groupby('WTeamID').size().reset_index(name='WinCount')

# Compute Team Loss Counts
team_loss_counts = regular_season.groupby('LTeamID').size().reset_index(name='LossCount')

# Merge to form basic team strength metric
team_strength = pd.merge(team_win_counts, team_loss_counts, left_on='WTeamID', right_on='LTeamID', how='outer').fillna(0)
team_strength['TeamID'] = team_strength['WTeamID'].fillna(team_strength['LTeamID'])

# Drop redundant columns
team_strength = team_strength[['TeamID', 'WinCount', 'LossCount']]

# Calculate win percentage
team_strength['WinPct'] = team_strength['WinCount'] / (team_strength['WinCount'] + team_strength['LossCount'])

# Preview engineered features
print(team_strength.sort_values(by='WinPct', ascending=False).head(10))
