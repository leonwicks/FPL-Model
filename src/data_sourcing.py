from concurrent.futures import ThreadPoolExecutor
import logging
import requests
import pandas as pd
from tqdm.auto import tqdm
import yaml

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

def process_fpl_data(df_players, df_teams, df_positions, df_points):
    """Reduce the number of columns to only retain desired features. Merge all datasets"""
    # specify columns to persist for each dataframe
    feature_config_path = r'meta\configuration\config.yml'
    with open(feature_config_path, encoding="utf-8") as f:
        feature_config = yaml.safe_load(f)

    # players features
    players_cols_to_keep = feature_config['players_table_features']

    # teams features
    teams_cols_to_keep = feature_config['teams_table_features']

    # positions features
    positions_cols_to_keep = feature_config['positions_table_features']

    # points features
    points_cols_to_keep = feature_config['points_table_features']

    df_merged = df_players[players_cols_to_keep].merge(
        df_teams[teams_cols_to_keep],
        how='inner',
        left_on='team_code',
        right_on='code',
        suffixes=['','_teams'],
    ).merge(
        df_positions[positions_cols_to_keep],
        how='inner',
        left_on='element_type',
        right_on='id',
        suffixes=['','_pos'],
    ).merge(
        df_points[points_cols_to_keep],
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
    return df_merged

def fetch_fpl_data():
    df_players, df_teams, df_positions, df_points = get_fpl_data()
    df_merged = process_fpl_data(df_players, df_teams, df_positions, df_points)
    return df_merged

if __name__ == '__main__':
    df = fetch_fpl_data()
    print(df.head())