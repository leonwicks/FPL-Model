import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime, timedelta
from tqdm.auto import tqdm
tqdm.pandas()

def _get_gameweek_history(player_id):
    url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
    response = requests.get(url).json()
    df_history = pd.json_normalize(response['history'])
    return df_history

def get_fpl_data():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url).json()
    
    df_players = pd.json_normalize(response['elements'])
    df_teams = pd.json_normalize(response['teams'])
    df_positions = pd.json_normalize(response['element_types'])

    df_master = df_players.merge(
        df_positions,
        how='inner',
        left_on='element_type',
        right_on='id',
        suffixes=['_player','_position']
    ).merge(
        df_teams,
        how='inner',
        left_on='team',
        right_on='id',
        suffixes=['','_team']
    )

    print("Getting gameweek history.")
    df_points = df_master['id_player'].progress_apply(_get_gameweek_history)
    df_points = pd.concat(df for df in df_points)

    
    df_points = df_master[['id_player', 
                           'web_name', 
                           'name', 
                           'strength',
                           'chance_of_playing_next_round',
                           'singular_name_short']].merge(
        df_points,
        left_on='id_player',
        right_on='element'
    )
    return df_players, df_teams, df_points

def get_understat_data():
    # Step 1: Fetch the website
    url = f'https://understat.com/league/EPL/2024'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Step 2: Find the specific <script> tag containing the JSON data
    scripts = soup.find_all('script')
    json_data = None

    # Step 3: Look for the script containing the teamsData
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

def _remap_team_names_row(row,team_name_col):
    teams_to_rename = {
        'Manchester City':'Man City',
        'Manchester United':'Man Utd',
        'Newcastle United':'Newcastle',
        'Nottingham Forest':"Nott'm Forest",
        'Tottenham':'Spurs',
        'Wolverhampton Wanderers':'Wolves',
    }

    if row[team_name_col] in (teams_to_rename.keys()):
        row[team_name_col] = teams_to_rename[row[team_name_col]]

    return row

def _remap_team_names_df(df,team_name_col):
    df[team_name_col] = df.index
    df = df.apply(_remap_team_names_row, axis=1)
    return df

def calc_upcoming_fixture_difficulty(df_dates, df_teams):
    current_dt = datetime.now()
    current_dt_plus_3wks = current_dt + timedelta(days=30)

    df_upcoming = df_dates[(pd.to_datetime(df_dates['datetime']) >= current_dt) & (pd.to_datetime(df_dates['datetime']) < current_dt_plus_3wks)]
    
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
    df_upcoming_combined = _remap_team_names_df(df_upcoming_combined,'team_name')
    return df_upcoming_combined

def calc_exp_goals_conceded(df_dates):
    current_dt = datetime.now()

    df_dates = df_dates[pd.to_datetime(df_dates['datetime']) <= current_dt]
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
    df_xgc = _remap_team_names_df(df_xgc,'team_name')
    return df_xgc

def group_points_data(df_points, df_xgc, df_upcoming):
    df_points = df_points.merge(df_xgc[['team_name','team_xgc_per_game']],
                                how='inner',
                                left_on='name',
                                right_on='team_name',
                                suffixes=['','_xgc'])

    df_points = df_points.merge(df_upcoming[['team_name','mean_strength']],
                                how='inner',
                                left_on='team_name',
                                right_on='team_name',
                                suffixes=['','_upcoming'])
    
    df_points = df_points[df_points['chance_of_playing_next_round'] >= 75]

    cols_to_group = [
        'id_player',
        'singular_name_short',
    ]

    points_cols_to_keep = [
        'id_player', 
        'expected_goals', 
        'expected_assists',
        'expected_goals_conceded', 
        'minutes',
        'value',
        'singular_name_short',
        'team_xgc_per_game',
        'mean_strength',
    ]

    df_points = df_points[points_cols_to_keep]

    for col in df_points:
        if col in cols_to_group:
            continue
        elif col in ['minutes']:
            continue
        else:
            df_points[col] = df_points[col].astype(float)

    df_points_grp = df_points.groupby(by=cols_to_group,as_index=False).mean()
    return df_points_grp

def add_additional_cols(df_exp_points, df_players, df_teams):
    df_exp_points = df_exp_points.merge(
                                        df_players[['id','now_cost','team','web_name']],
                                        how='left',
                                        left_on='id_player',
                                        right_on='id',
                                        ).merge(
                                        df_teams[['id','name']],
                                        how='left',
                                        left_on='team',
                                        right_on='id'
                                        )
    return df_exp_points


