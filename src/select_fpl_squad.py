import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, PULP_CBC_CMD

from data_sourcing import fetch_fpl_data
from feature_engineering import engineer_features

import logging
logging.basicConfig(level=logging.INFO)

def select_fpl_squad(df,
                     metric,
                     num_gks=2,
                     num_defs=5,
                     num_mids=5,
                     num_atts=3,
                     num_mngs=0,
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

    cols_to_keep = [
        'id_player',
        'web_name',
        'now_cost',
        'player_position',
        'team_name',
        'chance_of_playing_next_round',
        metric,
    ]

    df = df[cols_to_keep].drop_duplicates()

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
    positions = {'GKP': num_gks, 'DEF': num_defs, 'MID': num_mids, 'FWD': num_atts, 'MNG': num_mngs}
    for pos, count in positions.items():
        pos_ids = df[df['player_position'] == pos].index.tolist()
        prob += lpSum(x[i] for i in pos_ids) == count, f"Total_{pos.upper()}"

    # Team constraints: no more than 3 players from each team
    for team in df['team_name'].unique():
        team_ids = df[df['team_name'] == team].index.tolist()
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
    else:
        logging.info("Optimal solution found.")


    # Get the selected players
    selected_ids = [i for i in player_ids if x[i].varValue == 1]
    selected_squad = df.loc[selected_ids].reset_index(drop=True)

    return selected_squad

if __name__ == '__main__':
    num_gws = 7
    metric = f'mean_ppg_{num_gws}'

    df = fetch_fpl_data()
    df = engineer_features(df, num_gws)
    squad = select_fpl_squad(df, metric)
    print(squad.sort_values(by='player_position'))