### FPL Bot v1r2
#
# Revisions include:
# - Move FPL API data merge out of get_fpl_data
# - Reduce functionality of filter_dataframe
# - Add upcoming ficture difficulty and team xgc to df_teams instead of making new dataframes
# - Add number days to look forward as argument in calc_upcoming_fixture_difficulty
# - Add number weeks to look back as argument in calc_exp_goals_conceded
# - Moved all data transformations into a transform_data function


import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime, timedelta
import math
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, PULP_CBC_CMD
from tabulate import tabulate
from tqdm.auto import tqdm
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
    response = requests.get(url).json()
    df_history = pd.json_normalize(response['history'])
    return df_history

def get_fpl_data():
    """
    Load player, team, position, and points history data from the FPL API.

    Outputs:
    - df_players: pandas dataframe containing player data
    - df_teams: pandas dataframe containing team data
    - df_positions: pandas dataframe containing position data
    - df_points: pandas dataframe containing points history data
    """
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url).json()
    
    df_players = pd.json_normalize(response['elements'])
    df_teams = pd.json_normalize(response['teams'])
    df_positions = pd.json_normalize(response['element_types'])

    df_points = df_players['id'].progress_apply(get_gameweek_history)
    df_points = pd.concat(df for df in df_points)
    return df_players, df_teams, df_positions, df_points

def get_understat_data():
    """
    Load data from the understat website to calculate team expected goals conceded and upcoming fixture difficulties.

    Outputs:
    - df_dates: pandas dataframe containing understat data
    """
    # Step 1: Fetch the website
    url = f'https://understat.com/league/EPL/2024'
    response = requests.get(url)
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

    if row['team_name'] in (teams_to_rename.keys()):
        row['team_name'] = teams_to_rename[row['team_name']]

    return row

def calc_upcoming_fixture_difficulty(df_dates, df_teams, max_date=None):
    """
    Calculate the mean fixture difficulty for upcoming 30 days for each team.

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

def filter_dataframe(df,filters={}):
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

    # if row['minutes'] > 0:
    #     if row['minutes'] >= 60:
    #         expected_mins_points = 2
    #     else:
    #         expected_mins_points = 1
    # else:
    #     row['expected_points'] = 0
    #     return row

    minutes_multiplier = row['minutes'] / 90

    expected_goal_points = row['expected_goals'] * points_by_pos[row['player_position']]['goal']
    expected_ass_points = row['expected_assists'] * points_by_pos[row['player_position']]['ass']
    expected_cs_points = math.exp(-row['team_xgc_per_game']) * points_by_pos[row['player_position']]['cs']
    expected_gc_points_lost = row['expected_assists'] * points_by_pos[row['player_position']]['gc']

    if row['mean_strength'] >= 3.5:
        fixture_multiplier = 0.9
    elif row['mean_strength'] <= 2.5:
        fixture_multiplier = 1.1
    else:
        fixture_multiplier = 1
    
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
        print("No optimal solution found.")
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
    squad = squad.sort_values(by=['player_position','teams_name','player_name']).round(2)
    print(tabulate(squad, 
                   headers='keys', 
                   tablefmt='psql'
                   ))
    
    current_date = datetime.now().date()
    squad.to_csv(r'C:\Users\Leon\Documents\Football Modeling\FPL Model\fpl bot draft 2\squad_{}.csv'.format(current_date))

def transform_data(
        df_points,
        df_players,
        df_teams,
        df_positions,
        df_dates,
        rounds_to_sub=5, 
        min_chance_of_playing=75,
        ):
    # Step 1. Filter data
    min_round = df_points['round'].max() - rounds_to_sub
    
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

    # Step 4. Group Data

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

    df_grp = df.groupby(by=cols_to_group,as_index=False).mean()

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

    metric = 'expected_points'
    selected_squad = select_fpl_squad(df_exp_points, metric)
    save_selected_squad(selected_squad)

main()