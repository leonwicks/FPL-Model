import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import re

# Step 1: Fetch the webpage content
url = "https://understat.com/league/EPL"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Step 2: Find the JavaScript containing the team data
scripts = soup.find_all('script')
pattern = re.compile(r"teamsData\s+=\s+JSON\.parse\('([^']+)'\);")
json_data = None

for script in scripts:
    if 'teamsData' in script.text:
        match = pattern.search(script.text)
        if match:
            json_data = match.group(1).encode().decode('unicode_escape')
            break

# Step 3: Convert the JSON data to a DataFrame
if json_data:
    data = json.loads(json_data)
    teams_data = []

    for team_id, team_stats in data.items():
        team_data = {
            'team': team_stats['title'],  # Team name
            'id': team_id,  # Team ID
            'history': team_stats['history']  # List of game stats
        }
        teams_data.append(team_data)

    # Create a DataFrame
    df = pd.DataFrame(teams_data)
    
    # Step 4: Explode the 'history' column to have each game in its own row
    df_exploded = df.explode('history').reset_index(drop=True)
    
    # Step 5: Normalize the 'history' column into separate columns
    df_normalized = pd.concat([df_exploded.drop(columns=['history']), df_exploded['history'].apply(pd.Series)], axis=1)
    
    # Display the normalized DataFrame
    print(df_normalized.head())
else:
    print("Could not find the team data.")
