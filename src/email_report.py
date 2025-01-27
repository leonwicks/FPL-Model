from data_sourcing import fetch_fpl_data
from feature_engineering import engineer_features
from select_fpl_squad import select_fpl_squad

import smtplib
from email.mime.text import MIMEText
import yaml
import logging


def email_squad(squad, metric):
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
            mean_ppg = round(row[metric], 2)

            player_str = f"| {name} | {team} |\n| Cost: {cost}m | Points per Game: {mean_ppg} |\n\n"
            final_str += player_str
        final_str += buffer_str

    email_details_path = 'meta\configuration\email_credentials.yml'
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

if __name__ == '__main__':
    num_gws_to_check = 7
    metric = f'mean_ppg_{num_gws_to_check}'

    df = fetch_fpl_data()
    df = engineer_features(df, num_gws_to_check)
    squad = select_fpl_squad(df=df, metric=f'mean_ppg_{num_gws_to_check}')
    email_squad(squad, metric)