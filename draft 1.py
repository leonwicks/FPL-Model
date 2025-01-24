import requests
import pandas as pd

def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    return response

response = fetch_fpl_data()
print(response)