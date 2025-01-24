### FPL Bot v2r0
#
# FPL Bot V2r0 is a form based bot, selecting the team with the highest points per game 
# over the previous x weeks, where x is an input parameter defaulting to 3.


from concurrent.futures import ThreadPoolExecutor
import logging
import requests
import pandas as pd
import sys
from datetime import datetime
from tabulate import tabulate
from tqdm.auto import tqdm
import smtplib
from email.mime.text import MIMEText
import yaml

sys.path.append('src\models')
from select_fpl_squad import select_fpl_squad

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

def calc_points_per_game(df, num_gws=3):
    # filter for last x gameweeks
    df = df[df['round'] > df['round'].max() - num_gws]

    # group by player
    df_ppg = df.groupby(by='element', as_index=False)['total_points'].mean()
    return df_ppg

def process_fpl_data(df_players, 
                     df_positions, 
                     df_teams,
                     df_ppg,
                    ):
    """Reduce the number of columns to only retain desired features. Merge all datasets"""
    # specify columns to persist for each dataframe
    feature_config_path = r'meta\v1r0\config.yml'
    with open(feature_config_path, encoding="utf-8") as f:
        feature_config = yaml.safe_load(f)
    
    # df_players
    players_cols_to_keep = feature_config['players_table_features']

    # df_positions
    positions_cols_to_keep = feature_config['positions_table_features']

    # df_teams
    teams_cols_to_keep = feature_config['teams_table_features']

    # merge reduced dataframes
    df = df_players[players_cols_to_keep].merge(
        df_positions[positions_cols_to_keep],
        how='inner',
        left_on='element_type',
        right_on='id',
        suffixes=['','_pos']
    ).merge(
        df_teams[teams_cols_to_keep],
        how='inner',
        left_on='team_code',
        right_on='code',
        suffixes=['','_team']
    ).merge(
        df_ppg,
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
        'total_points':'mean_ppg',      
    }

    df.rename(columns=cols_to_rename, inplace=True)
    return df

def save_selected_squad(squad):
    """
    Print selected squad using tabulate and save to csv file

    Inputs:
        - squad: pandas dataframe containing 15 man squad
    """
    cols_to_print = [
        'web_name',
        'team_name',
        'mean_ppg',
        'player_position',
        'now_cost',
        'chance_of_playing_next_round',

    ]
    squad['player_position'] = pd.Categorical(squad['player_position'], ['GKP','DEF','MID','FWD'])
    squad = squad.sort_values(by=['player_position','mean_ppg']).round(2)
    print(tabulate(squad[cols_to_print], 
                   headers='keys',
                   tablefmt='psql'
                   ))
    
    current_date = datetime.now().date()
    
    logging.info("Saving squad.")
    save_path = r"meta\squads\squad_{}.csv".format(current_date)
    squad.to_csv(save_path)
    
def email_squad(squad):
    """Email the selected squad from a deignated gmail account."""
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    final_str = ""
    buffer_str = "\n"

    for pos in positions:
        df_pos = squad[squad['player_position'] == pos]
        intro_str = f"The {pos}s selected are:\n"
        final_str += intro_str
        for _, row in df_pos.iterrows():
            name = row['web_name']
            team = row['team_name']
            cost = row['now_cost'] / 10
            mean_ppg = round(row['mean_ppg'], 2)

            player_str = f"| {name} | {team} |\n| Cost: {cost}m | Points per Game: {mean_ppg} |\n\n"
            final_str += player_str
        final_str += buffer_str

    email_details_path = '.gitignore\email credentials.yml'
    with open(email_details_path, encoding="utf-8") as f:
        email_details = yaml.safe_load(f)

    subject = "Sven Botman's FPL Team of the Week"
    sender = email_details['sender_email']
    recipients = email_details['recipient_email']
    pword = email_details['password']


    msg = MIMEText(final_str)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(sender, pword)
        smtp_server.sendmail(sender, recipients, msg.as_string())
    
    logging.info("Email sent!")

def main():
    df_players, df_teams, df_positions, df_points = get_fpl_data()
    df_ppg = calc_points_per_game(df_points,num_gws=7)
    df = process_fpl_data(df_players, df_positions, df_teams, df_ppg)
    squad = select_fpl_squad(df=df, metric='mean_ppg')
    save_selected_squad(squad)
    # email_squad(squad)

main()