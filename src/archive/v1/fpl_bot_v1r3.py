### FPL Bot v1r3
#
# Revisions include:
# - Use multithreading to speed up FPL API calls
# - Add error handling to API responses
# - Modified fixture difficulty multiplier to be continuous
# - Added logs
# - Minor changes in accordance with pylint reccomendations


import json
from datetime import datetime, timedelta
import math
from concurrent.futures import ThreadPoolExecutor
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, PULP_CBC_CMD
from tabulate import tabulate
from tqdm.auto import tqdm

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
    """
    Load data from the understat website to calculate team expected goals conceded and upcoming fixture difficulties.

    Outputs:
    - df_dates: pandas dataframe containing understat data
    """
    logging.info("Fetching Understat data.")

    # Step 1: Fetch the website
    url = 'https://understat.com/league/EPL/2024'
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Step 2: Find the specific <script> tag containing the JSON data
    scripts = soup.find_all('script')
    json_data = None

    # Step 3: Look for the script containing the datesData
    for script in scripts:
        if 'datesData' in script.text:
            # Extract the content of the script tag as text
            json_text = script.text
            # Step 4: Isolate the JSON data by splitting the string
            json_text = json_text.split("JSON.parse('")[1].split("')")[0]
            # Step 5: Decode the string by replacing escape characters
            json_text = json_text.encode('utf-8').decode('unicode_escape')
            # Step 6: Load the decoded text as a Python dictionary
            json_data = json.loads(json_text)
            break

    data = []
    for match in json_data:
        data.append({
            'home_team': match['h']['title'],
            'away_team': match['a']['title'],
            'home_goals': match['goals']['h'],
            'away_goals': match['goals']['a'],
            'xG_home': match['xG']['h'],
            'xG_away': match['xG']['a'],
            'datetime': match['datetime']
        })

    df_dates = pd.DataFrame(data)
    return df_dates

def _remap_team_names(row):
    """
    Rename team names from understat to match FPL API format
    """
    teams_to_rename = {
        'Manchester City':'Man City',
        'Manchester United':'Man Utd',
        'Newcastle United':'Newcastle',
        'Nottingham Forest':"Nott'm Forest",
        'Tottenham':'Spurs',
        'Wolverhampton Wanderers':'Wolves',
    }

    if row['team_name'] in teams_to_rename.keys():
        row['team_name'] = teams_to_rename[row['team_name']]

    return row

def calc_upcoming_fixture_difficulty(df_dates, df_teams, max_date=None):
    """
    Calculate the mean fixture
      difficulty for upcoming 30 days for each team.

    Inputs: 
    - df_dates: pandas dataframe containing fixture data (inc. past and future)
    - df_teams: pandas dataframe containing team data

    Outputs:
    - df_teams: pandas dataframe containing df_teams data
    """
    current_dt = datetime.now()
    if not max_date:
        max_date = current_dt + timedelta(days=30)

    df_upcoming = df_dates[(pd.to_datetime(df_dates['datetime']) >= current_dt) & (pd.to_datetime(df_dates['datetime']) < max_date)]

    df_upcoming = df_upcoming.merge(df_teams[['name','strength']],
                                    how='left',
                                    left_on='home_team',
                                    right_on='name').rename(columns={'strength':'home_strength'})

    df_upcoming = df_upcoming.merge(df_teams[['name','strength']],
                                    how='left',
                                    left_on='away_team',
                                    right_on='name').rename(columns={'strength':'away_strength'})

    df_upcoming_home = df_upcoming.groupby(by=['home_team']).agg({
        'away_team':'nunique',
        'away_strength':'sum',
    })

    df_upcoming_away = df_upcoming.groupby(by=['away_team']).agg({
        'home_team':'nunique',
        'home_strength':'sum',
    })

    df_upcoming_combined = df_upcoming_home.merge(df_upcoming_away,
                                         how='outer',
                                         left_index=True,
                                         right_index=True
                                         ).fillna(0)

    df_upcoming_combined['mean_strength'] = (df_upcoming_combined['home_strength'] + df_upcoming_combined['away_strength']) / (df_upcoming_combined['home_team'] + df_upcoming_combined['away_team'])
    df_upcoming_combined['team_name'] = df_upcoming_combined.index
    df_upcoming_combined = df_upcoming_combined.apply(_remap_team_names, axis=1)

    df_teams = df_teams.merge(
        df_upcoming_combined[['team_name','mean_strength']],
        how='inner',
        left_on='name',
        right_on='team_name',
        suffixes=['','_upcoming']
    )
    return df_teams

def calc_exp_goals_conceded(df_dates, df_teams, min_date=None):
    """
    Calculate the expected goals conceded for each team for all games in points dataset.

    Inputs:
    - df_dates: pandas dataframe containing understat fixture data (inc. historic and past)

    Outputs: 
    - df_teams: pandas dataframe containing team data
    """
    current_dt = datetime.now()
    if not min_date:
        min_date = current_dt - timedelta(weeks=5)

    df_dates = df_dates[(pd.to_datetime(df_dates['datetime']) <= current_dt) & (pd.to_datetime(df_dates['datetime']).dt.date >= min_date)]
    df_dates['xG_home'] = df_dates['xG_home'].astype(float)
    df_dates['xG_away'] = df_dates['xG_away'].astype(float)

    df_xgc_h = df_dates.groupby(by=['home_team']).agg({
        'away_team':'nunique',
        'xG_away':'mean',
    })

    df_xgc_a = df_dates.groupby(by=['away_team']).agg({
        'home_team':'nunique',
        'xG_home':'mean',
    })

    df_xgc = df_xgc_h.merge(df_xgc_a,
                            how='inner',
                            left_index=True,
                            right_index=True)

    df_xgc['total_xgc'] = df_xgc['away_team'] * df_xgc['xG_away'] + df_xgc['home_team'] * df_xgc['xG_home']
    df_xgc['team_xgc_per_game'] = df_xgc['total_xgc'] / (df_xgc['away_team'] + df_xgc['home_team'])
    df_xgc['team_name'] = df_xgc.index
    df_xgc = df_xgc.apply(_remap_team_names, axis=1)

    df_teams = df_teams.merge(
        df_xgc[['team_name','team_xgc_per_game']],
        how='inner',
        left_on='name',
        right_on='team_name',
        suffixes=['','_xgc']
    )
    return df_teams

def filter_dataframe(df,filters=None):
    """
    Filters dataframe based on a dictionary of filters, where the keys are column names and the values are filter values (>= filter applied).
    """
    if not filters:
        return df

    for col in filters.keys():
        df = df[df[col] >= filters[col]]
    return df

def _calculate_exp_points(row):
    """
    Calculate expected points for each player given performance in previous weeks.

    Inputs: 
        - row: pandas dataframe row

    Outputs: 
        - row: pandas dataframe row (inc. expected points calculations)
    """
    points_by_pos = {
        'GKP':{'goal':10, 'ass':3, 'cs':4, 'gc':-0.5},
        'DEF':{'goal':6, 'ass':3, 'cs':4, 'gc':-0.5},
        'MID':{'goal':5, 'ass':3, 'cs':1, 'gc':0},
        'FWD':{'goal':4, 'ass':3, 'cs':0, 'gc':0}
    }

    minutes_multiplier = row['minutes'] / 90

    expected_goal_points = row['expected_goals'] * points_by_pos[row['player_position']]['goal']
    expected_ass_points = row['expected_assists'] * points_by_pos[row['player_position']]['ass']
    expected_cs_points = math.exp(-row['team_xgc_per_game']) * points_by_pos[row['player_position']]['cs']
    expected_gc_points_lost = row['expected_assists'] * points_by_pos[row['player_position']]['gc']

    fixture_multiplier = 1 + (3 - row['mean_strength']) * 0.1

    row['expected_points'] = fixture_multiplier * minutes_multiplier * (expected_goal_points + expected_ass_points + expected_cs_points + expected_gc_points_lost)
    return row

def apply_exp_goals_calcs(df_points):
    """
    Apply expected goals calculations to points dataframe
    
    Inputs: 
        - df_points: pandas dataframe containing points data
        
    Outputs: 
        - df_exp_points: pandas dataframe containing expected points data
    """
    print("Calculating expected points.")
    df_exp_points = df_points.progress_apply(_calculate_exp_points, axis=1)
    return df_exp_points

def select_fpl_squad(df,
                     metric,
                     num_gks=2,
                     num_defs=5,
                     num_mids=5,
                     num_atts=3,
                     max_value=1000):
    """
    Selects the optimal 15-man squad based on the given constraints.
    
    Inputs:
        - df: pandas dataframe containing player data
        - metric: column name to be optimised
        - num_gks: number of goalkeepers to be selected in squad
        - num_defs: number of defenders to be selected in squad
        - num_mids: number of midfielders to be selected in squad
        - num_atts: number of attackers to be selected in squad
        - max_value: maximum team value (multiplied by 10 so no decimal points)
    
    Returns:
        - selected_squad: pandas dataframe containing the selected squad
    """
    logging.info("Selecting optimal FPL squad.")

    # Reset index to ensure it ranges from 0 to N-1
    df = df.reset_index(drop=True)
    df.set_index('id_player',inplace=True)
    player_ids = df.index

    # Convert 'value' and 'P' columns to numeric
    df['now_cost'] = pd.to_numeric(df['now_cost'])
    df[metric] = pd.to_numeric(df[metric])

    # Define the LP problem
    prob = LpProblem("FPL_Squad_Selection", LpMaximize)

    # Define binary decision variables for each player
    x = LpVariable.dicts('x', player_ids, cat='Binary')

    # Objective function: maximize total points
    prob += lpSum(df.loc[i, metric] * x[i] for i in player_ids), "Total_Points"

    # Total value constraint
    prob += lpSum(df.loc[i, 'now_cost'] * x[i] for i in player_ids) <= max_value, "Total_value"

    # Position constraints
    positions = {'GKP': num_gks, 'DEF': num_defs, 'MID': num_mids, 'FWD': num_atts}
    for pos, count in positions.items():
        pos_ids = df[df['player_position'] == pos].index.tolist()
        prob += lpSum(x[i] for i in pos_ids) == count, f"Total_{pos.upper()}"

    # Team constraints: no more than 3 players from each team
    for team in df['teams_name'].unique():
        team_ids = df[df['teams_name'] == team].index.tolist()
        prob += lpSum(x[i] for i in team_ids) <= 3, f"Team_{team}"

    # Add constraints to ensure each player is selected at most once
    for i in player_ids:
        prob += x[i] <= 1, f"Select_{i}_At_Most_Once"

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=0))

    # Check if an optimal solution was found
    if LpStatus[prob.status] != 'Optimal':
        logging.info("No optimal solution found.")
        return None

    # Get the selected players
    selected_ids = [i for i in player_ids if x[i].varValue == 1]
    selected_squad = df.loc[selected_ids].reset_index(drop=True)

    return selected_squad

def save_selected_squad(squad):
    """
    Print selected squad using tabulate and save to csv file

    Inputs:
        - squad: pandas dataframe containing 15 man squad
    """
    squad['player_position'] = pd.Categorical(squad['player_position'], ['GKP','DEF','MID','FWD'])
    squad = squad.sort_values(by=['player_position','expected_points']).round(2)
    print(tabulate(squad, 
                   headers='keys',
                   tablefmt='psql'
                   ))

    current_date = datetime.now().date()
    
    logging.info("Saving squad.")
    squad.to_csv(r"C:\Users\Leon\Documents\Football Modeling\FPL Model\fpl bot draft 2\squad_{}.csv".format(current_date))

def transform_data(
        df_points,
        df_players,
        df_teams,
        df_positions,
        df_dates,
        rounds_to_sub=5, 
        min_chance_of_playing=75,
        ):
    """
    Applies a variety of data transformations. Follows these steps:

    1. Filter data
    2. Calculate new columns
    3. Merge data
    4. Aggregate data
    5. Calculate expected points

    Inputs:
        - df_points: pandas dataframe containing historic points data
        - df_players: pandas dataframe containing player data
        - df_teams: pandas dataframe containing team data
        - df_positions: pandas dataframe containing position data
        - df_dates: pandas dataframe containing understat fixture data
        - rounds_to_sub: number of most recent rounds to focus on
        - min_chance_of_playing: minimum chance of playing to consider

    Outputs:
        - df_exp_points: pandas dataframe containing aggregated expected points data
    """
    logging.info("Transforming data.")

    # Step 1. Filter data
    min_round = df_points['round'].max() - rounds_to_sub + 1

    points_filters = {
        'round':min_round,
    }

    players_filters = {
        'chance_of_playing_next_round':min_chance_of_playing,
    }

    df_points = filter_dataframe(df_points,points_filters)
    df_players = filter_dataframe(df_players,players_filters)
    

    # Step 2. Calculate new columns
    xgc_min_week = datetime.strptime(df_points['kickoff_time'].str[:10].min(), '%Y-%m-%d').date()

    df_teams = calc_upcoming_fixture_difficulty(df_dates, df_teams)
    df_teams = calc_exp_goals_conceded(df_dates, df_teams, min_date=xgc_min_week)

    # Step 3. Merge dataframes
    df = df_points.merge(
        df_players,
        how='inner',
        left_on='element',
        right_on='id',
        suffixes=['','_player']
    ).merge(
        df_teams,
        how='inner',
        left_on='team',
        right_on='id',
        suffixes=['','_team']
    ).merge(
        df_positions,
        how='inner',
        left_on='element_type',
        right_on='id',
        suffixes=['','_pos']
    )
        
    cols_to_rename = {
        'element':'id_player',
        'web_name':'player_name',
        'name':'teams_name',
        'singular_name_short':'player_position',
    }

    df.rename(columns=cols_to_rename,inplace=True)

    # Step 4. Aggregate Data

    cols_to_keep = [
        'id_player', 
        'player_name',
        'teams_name',
        'expected_goals', 
        'expected_assists',
        'expected_goals_conceded', 
        'minutes',
        'player_position',
        'team_xgc_per_game',
        'mean_strength',
        'now_cost',
        'total_points',
    ]

    df = df[cols_to_keep]

    cols_to_group = [
        'id_player',
        'player_position',
        'now_cost',
        'player_name',
        'teams_name',
    ]

    for col in df:
        if col in cols_to_group:
            continue
        elif col in ['minutes']:
            continue
        else:
            df[col] = df[col].astype(float)

    df_grp = df.groupby(by=cols_to_group,as_index=False).median()

    # Step 5. Calculate Expected Points
    df_exp_points = apply_exp_goals_calcs(df_grp)

    return df_exp_points

def main():
    """Run FPL Bot"""
    df_players, df_teams, df_positions, df_points = get_fpl_data()
    df_dates = get_understat_data()

    df_exp_points = transform_data(
        df_points,
        df_players,
        df_teams,
        df_positions,
        df_dates,
        rounds_to_sub=5,
        min_chance_of_playing=75,
        )

    metric = 'total_points'
    selected_squad = select_fpl_squad(df_exp_points, metric)
    save_selected_squad(selected_squad)

main()
