from data_sourcing import fetch_fpl_data
import pandas as pd

def calc_points_per_game(df, col, num_gws=3, override_col_name=None):
    """
    Calculate the mean of a column per game for each player over a given number of gameweeks.
    """
    # Ensure 'round' column is numeric for arithmetic
    if 'round' in df.columns:
        df['round'] = pd.to_numeric(df['round'], errors='coerce')
        print(f"[DEBUG] 'round' dtype in calc_points_per_game:", df['round'].dtype)
    # filter for last x gameweeks
    df_temp = df[df['round'] > df['round'].max() - num_gws]
    # group by player
    df_ppg = df_temp.groupby(by='element', as_index=False)[col].mean()
    if override_col_name:
        df_ppg.rename(columns={col:override_col_name}, inplace=True)
    else:
        df_ppg.rename(columns={col:f'mean_{col}_ppg_{num_gws}'}, inplace=True)
    # merge form column back to original dataframe
    df = df.merge(
        df_ppg,
        how='inner',
        left_on='element',
        right_on='element',
        )
    return df

def calc_rolling_mean(df, col, period):
    """Calculates the rolling mean of a column over a provided period."""
    df_temp = (df.sort_values(by=['id_player', 'round'], ascending=True)  # Ensure proper ordering
                 .groupby(by='id_player', as_index=False)[col]
                 .rolling(window=period)  # Specify window size and minimum periods
                 .mean()
                 .merge(
                    df[['id_player', 'round']]
                      .sort_values(by=['id_player', 'round'])
                      .drop('id_player', axis=1),
                    how='inner',
                    left_index=True,
                    right_index=True,
            )
                 .rename(columns={col:f'{col}_rolling_mean_{period}'})
                 .reset_index(drop=True)
    )

    df_rm = df.merge(df_temp, how='inner', left_on=['id_player', 'round'], right_on=['id_player', 'round'])
    return df_rm

def calc_rolling_mean_w_shift(df, target_column, window_size, shift_val=1):
    """
    Adds a new column with the rolling mean of a specified column, calculated over a 
    specified number of rows, partitioned by 'id_player', ordered by 'round', and shifted down by 1.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the column for which to calculate the rolling mean.
    - window_size (int): The number of rows to use for the rolling mean.

    Returns:
    - pd.DataFrame: A new DataFrame with the rolling mean column added.
    """
    # Validate required columns
    if 'id_player' not in df.columns or 'round' not in df.columns:
        raise ValueError("The DataFrame must contain 'id_player' and 'round' columns.")

    # Ensure the DataFrame is sorted by 'id_player' and 'round'
    df = df.sort_values(by=['id_player', 'round'])

    # Calculate the rolling mean and shift it down by 1
    rolling_column_name = f'rolling_mean_{target_column}_{window_size}'
    df[rolling_column_name] = (
        df.groupby('id_player', group_keys=False)
        .apply(lambda group: group[target_column].rolling(window=window_size, min_periods=window_size).mean().shift(shift_val))
    )

    return df

def calc_upcoming_fixture_difficulty(df):
    df_temp = df[df['round']]


def engineer_features(df, num_gws=3):
    df = calc_points_per_game(df, 'total_points', num_gws)
    df = calc_points_per_game(df, 'strength', num_gws)
    return df

if __name__ == '__main__':
    df = fetch_fpl_data()
    df = engineer_features(df)
    print(df.head())