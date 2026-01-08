# import requests

# def get_stock_data():
#     url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&outputsize=full&apikey=demo"
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         data = response.json()
#         last_refreshed = data["Meta Data"]["3. Last Refreshed"]
#         price = data["Time Series (5min)"][last_refreshed]["1. open"]
#         return price
#     else:
#         return None

# price = get_stock_data()
# symbol = "IBM"
# if price is not None:
#     print(f"{symbol}: {price}")
# else:
#     print("Failed to retrieve data.")

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json

# Your credentials from the Spotify Developer Dashboard
CLIENT_ID = '87793129d56f45fab6546f9fe0b735a0'
CLIENT_SECRET = '24635e683d784a289731b3f07c335338'
REDIRECT_URI = 'https://localhost:8000/callback'  # Make sure this matches the Redirect URI in your Spotify app settings
SCOPE = 'user-library-read playlist-modify-public' # Define the level of access you need

# Create a SpotifyOAuth object to handle authentication
sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope=SCOPE)

# Get the access token. This will open a browser window for you to log in and authorize the app
token_info = sp_oauth.get_access_token(as_dict=True)
token = token_info['access_token']

# Create the main Spotipy object
sp = spotipy.Spotify(auth=token)

# Optional: Print current user info to verify authentication
user = sp.current_user()
print(json.dumps(user, sort_keys=True, indent=4))
