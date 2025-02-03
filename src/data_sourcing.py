from concurrent.futures import ThreadPoolExecutor
import logging
import requests
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import json

logging.basicConfig(level=logging.INFO)
tqdm.pandas()

def get_gameweek_history(player_id):
    """
    Get gameweeek history from FPL API for given player ID
    
    Inputs: 
    - player_id: integer FPL ID of given player

    Outputs:
    - df_history: pandas dataframe containing points history for given player ID
    """
    url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.json_normalize(response.json()['history'])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for player {player_id}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

def get_all_gameweek_histories(player_ids):
    """Uses multithreading to speed up FPL API calls"""
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(get_gameweek_history, player_ids), total=len(player_ids)))
    return pd.concat(results, ignore_index=True)

def get_fpl_data():
    """
    Load player, team, position, and points history data from the FPL API.

    Outputs:
    - df_players: pandas dataframe containing player data
    - df_teams: pandas dataframe containing team data
    - df_positions: pandas dataframe containing position data
    - df_points: pandas dataframe containing points history data
    """
    logging.info("Fetching FPL data.")

    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FPL data : {e}")

    df_players = pd.json_normalize(response_json['elements'])
    df_teams = pd.json_normalize(response_json['teams'])
    df_positions = pd.json_normalize(response_json['element_types'])

    df_points = get_all_gameweek_histories(df_players['id'])
    return df_players, df_teams, df_positions, df_points

def get_understat_data():
    # Step 1: Fetch the website
    url = 'https://understat.com/league/EPL'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Step 2: Find the specific <script> tag containing the JSON data
    scripts = soup.find_all('script')
    dates_json_data = None

    # Step 3: Look for the script containing the teamsData
    for script in scripts:
        if 'datesData' in script.text:
            # Extract the content of the script tag as text
            dates_json_text = script.text
            # Step 4: Isolate the JSON data by splitting the string
            dates_json_text = dates_json_text.split("JSON.parse('")[1].split("')")[0]
            # Step 5: Decode the string by replacing escape characters
            dates_json_text = dates_json_text.encode('utf-8').decode('unicode_escape')
            # Step 6: Load the decoded text as a Python dictionary
            dates_json_data = json.loads(dates_json_text)
            break

    if dates_json_data:
        matches_by_date = {
            'match_id':[],
            'home_team_id':[],
            'home_team_long_title':[],
            'home_team_short_title':[],
            'away_team_id':[],
            'away_team_long_title':[],
            'away_team_short_title':[],
            'home_goals':[],
            'away_goals':[],
            'home_xg':[],
            'away_xg':[],
            'datetime':[],
        }

        for match in dates_json_data:
            matches_by_date['match_id'].append(match['id'])
            matches_by_date['home_team_id'].append(match['h']['id'])
            matches_by_date['home_team_long_title'].append(match['h']['title'])
            matches_by_date['home_team_short_title'].append(match['h']['short_title'])
            matches_by_date['away_team_id'].append(match['a']['id'])
            matches_by_date['away_team_long_title'].append(match['a']['title'])
            matches_by_date['away_team_short_title'].append(match['a']['short_title'])
            matches_by_date['home_goals'].append(match['goals']['h'])
            matches_by_date['away_goals'].append(match['goals']['a'])
            matches_by_date['home_xg'].append(match['xG']['h'])
            matches_by_date['away_xg'].append(match['xG']['a'])
            matches_by_date['datetime'].append(match['datetime'])

        df_dates = pd.DataFrame(matches_by_date)
        return df_dates
    else:
        print("Data could not be found")

def process_fpl_data(df_players, df_teams, df_positions, df_points, df_understat=None):
    """Reduce the number of columns to only retain desired features. Merge all datasets"""

    df_merged = df_players.merge(
        df_teams,
        how='inner',
        left_on='team_code',
        right_on='code',
        suffixes=['','_teams'],
    ).merge(
        df_positions,
        how='inner',
        left_on='element_type',
        right_on='id',
        suffixes=['','_pos'],
    ).merge(
        df_points,
        how='inner',
        left_on='id',
        right_on='element',
        suffixes=['_player','']
    )
    
    # rename columns for readability
    cols_to_rename = {
        'id':'id_player',
        'singular_name_short':'player_position',
        'name':'team_name',
    }

    df_merged.rename(columns=cols_to_rename, inplace=True)

    if df_understat is not None:
        team_name_mappings = {
            'Manchester City' : 'Man City', 
            'Manchester United' : 'Man Utd', 
            'Newcastle United' : 'Newcastle',
            'Nottingham Forest' : "Nott'm Forest", 
            'Tottenham' : 'Spurs', 
            'Wolverhampton Wanderers' : 'Wolves',
        }

        df_understat['home_team_long_title'] = np.where(df_understat['home_team_long_title'].isin(team_name_mappings.keys()), df_understat['home_team_long_title'].map(team_name_mappings), df_understat['home_team_long_title'])
        df_understat['away_team_long_title'] = np.where(df_understat['away_team_long_title'].isin(team_name_mappings.keys()), df_understat['away_team_long_title'].map(team_name_mappings), df_understat['away_team_long_title'])

        df_understat = df_understat.merge(
            df_teams[['name','strength']],
            how='inner',
            left_on=['away_team_long_title'],
            right_on=['name']
        ).rename(columns={'strength':'away_team_strength'})

        df_merged['kickoff_date'] = df_merged['kickoff_time'].str[:10]
        df_understat['kickoff_date'] = df_understat['datetime'].str[:10]

        df_merged = df_merged.merge(
            df_understat,
            how='inner',
            left_on=['team_name','kickoff_date'],
            right_on=['home_team_long_title','kickoff_date'],
            suffixes=['','_undst']
        )

    return df_merged

def fetch_fpl_data():
    df_players, df_teams, df_positions, df_points = get_fpl_data()
    df_understat = get_understat_data()
    df_merged = process_fpl_data(df_players, df_teams, df_positions, df_points, df_understat=df_understat)
    return df_merged

if __name__ == '__main__':
    df = fetch_fpl_data()
    print(df.shape)
    print(df.head())