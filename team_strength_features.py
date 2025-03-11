import pandas as pd

base_path = r'C:\Users\Lydiah\PyCharmProjects\PythonProject\PythonProject\March ML Mania 2025\data\march-machine-learning-mania-2025\\'

# Load Regular Season Compact Results
regular_season = pd.read_csv(base_path + 'MRegularSeasonCompactResults.csv')

# Compute Total Wins and Losses
team_wins = regular_season.groupby('WTeamID').size().reset_index(name='Wins')
team_losses = regular_season.groupby('LTeamID').size().reset_index(name='Losses')

# Merge win/loss stats
team_strength = pd.merge(team_wins, team_losses, left_on='WTeamID', right_on='LTeamID', how='outer').fillna(0)
team_strength['TeamID'] = team_strength['WTeamID'].combine_first(team_strength['LTeamID'])
team_strength = team_strength[['TeamID', 'Wins', 'Losses']]
team_strength['WinRatio'] = team_strength['Wins'] / (team_strength['Wins'] + team_strength['Losses'])

# ✅ Save team strength metrics
team_strength.to_csv(base_path + 'team_strength.csv', index=False)
print("✅ Team strength features saved!")


# 🔹 Compute Tournament Seed Difference
seeds_df = pd.read_csv(base_path + 'MNCAATourneySeeds.csv')

# Extract numerical seed value (e.g., "W01" → 1)
seeds_df['SeedNum'] = seeds_df['Seed'].apply(lambda x: int(x[1:3]))
seeds_df = seeds_df[['Season', 'TeamID', 'SeedNum']]

# ✅ Save tournament seeds
seeds_df.to_csv(base_path + 'team_seeds.csv', index=False)
print("✅ Tournament seed data saved!")


# 🔹 Compute Recent Performance (Last 10 Games Win Ratio)
regular_season['Last10Games'] = regular_season.groupby(['Season', 'WTeamID'])['WTeamID'].transform(lambda x: x.rolling(10, min_periods=1).count())
regular_season['WinRatioLast10'] = regular_season['Last10Games'] / 10

recent_performance = regular_season[['Season', 'WTeamID', 'WinRatioLast10']].drop_duplicates()

# ✅ Save recent performance metrics
recent_performance.to_csv(base_path + 'team_recent_performance.csv', index=False)
print("✅ Recent performance features saved!")
