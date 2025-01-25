from data_sourcing import fetch_fpl_data

def calc_points_per_game(df, num_gws=3):
    # filter for last x gameweeks
    df = df[df['round'] > df['round'].max() - num_gws]

    # group by player
    df_ppg = df.groupby(by='element')['total_points'].mean().to_frame()
    df_ppg.rename(columns={'total_points':f'mean_ppg_{num_gws}'}, inplace=True)

    # merge form column back to original dataframe
    df = df.merge(
        df_ppg,
        how='inner',
        left_on='element',
        right_index=True
        )
    return df

def engineer_features(df, num_gws=3):
    df = calc_points_per_game(df, num_gws)
    return df

if __name__ == '__main__':
    df = fetch_fpl_data()
    df = engineer_features(df)
    print(df.head())