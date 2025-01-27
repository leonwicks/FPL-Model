from data_sourcing import fetch_fpl_data

def calc_points_per_game(df, num_gws=3, override_col_name=None):
    """
    Calculate the mean points per game for each player over a given number of gameweeks.
    """
    # filter for last x gameweeks
    df_temp = df[df['round'] > df['round'].max() - num_gws]

    # group by player
    df_ppg = df_temp.groupby(by='element', as_index=False)['total_points'].mean()

    if override_col_name:
        df_ppg.rename(columns={'total_points':override_col_name}, inplace=True)
    else:
        df_ppg.rename(columns={'total_points':f'mean_ppg_{num_gws}'}, inplace=True)

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

def calc_rolling_mean_w_shift(df, col, period, shift_val=1):
    """
    Calculates the rolling mean of a column over a provided period, and shifts by 
    a specified number of rows.   
    """
    df_temp = (df.sort_values(by=['id_player', 'round'], ascending=True)  # Ensure proper ordering
                 .groupby(by='id_player', as_index=False)[col]
                 .rolling(window=period)  # Specify window size and minimum periods
                 .mean()
                 .shift(shift_val)
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


def engineer_features(df, num_gws=3):

    df = calc_points_per_game(df, num_gws)
    return df

if __name__ == '__main__':
    df = fetch_fpl_data()
    df = engineer_features(df)
    print(df.head())