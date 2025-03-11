import pandas as pd

# File paths
base_path = r'C:\Users\Lydiah\PyCharmProjects\PythonProject\PythonProject\March ML Mania 2025\data\march-machine-learning-mania-2025\\'

# Load tournament results
tourney_results = pd.read_csv(base_path + 'MNCAATourneyCompactResults.csv')
print("Tournament Results Preview:")
print(tourney_results.head(), "\n")

# Load tournament seeds
tourney_seeds = pd.read_csv(base_path + 'MNCAATourneySeeds.csv')
print("Tournament Seeds Preview:")
print(tourney_seeds.head(), "\n")

# Load team names
teams = pd.read_csv(base_path + 'MTeams.csv')
print("Teams Preview:")
print(teams.head())
