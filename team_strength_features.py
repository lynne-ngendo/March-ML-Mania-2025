import pandas as pd

base_path = r'C:\Users\Lydiah\PyCharmProjects\PythonProject\PythonProject\March ML Mania 2025\data\march-machine-learning-mania-2025\\'

# Load Regular Season Compact Results
regular_results = pd.read_csv(base_path + 'MRegularSeasonCompactResults.csv')

# Compute Team Wins and Losses
team_wins = regular_results.groupby('WTeamID').size().reset_index(name='Wins')
team_losses = regular_results.groupby('LTeamID').size().reset_index(name='Losses')

# Merge wins and losses
team_strength = pd.merge(team_wins, team_losses, left_on='WTeamID', right_on='LTeamID', how='outer').fillna(0)

# Finalize columns
team_strength['TeamID'] = team_strength['WTeamID'].combine_first(team_strength['LTeamID'])
team_strength = team_strength[['TeamID', 'Wins', 'Losses']]
team_strength['WinRatio'] = team_strength['Wins'] / (team_strength['Wins'] + team_strength['Losses'])

# 🚀 **Save the team strength features**
team_strength.to_csv(base_path + 'team_strength.csv', index=False)

print("✅ Team strength features saved successfully!")

