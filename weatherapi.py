# Adapted from - https://www.geeksforgeeks.org/python-find-current-weather-of-any-city-using-openweathermap-api/

import sys
import requests
import json

API_KEY = sys.argv[1]
CITY_NAME = sys.argv[2]

base_url = "http://api.openweathermap.org/data/2.5/weather?"

complete_url = (base_url +
                "appid=" + API_KEY + 
                "&q=" + CITY_NAME)

response = requests.get(complete_url)

response_json = response.json()

if response_json["cod"] != "404":
    data = response_json["main"]
    current_temp = data["temp"]
    current_rh = data["humidity"]
else:
    print("City not found")

print(f"City: {CITY_NAME}, Temp: {current_temp}, RH: {current_rh}")
