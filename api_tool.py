import requests


CODE_LOOKUP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_weather(city: str) -> str:
    geocode_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    )

    if geocode_response.status_code != 200:
        return f"Error: geocoding service returned status {geocode_response.status_code}."

    geocode_data = geocode_response.json().get("results")
    if not geocode_data:
        return f"City '{city}' not found."

    location = geocode_data[0]
    latitude = location["latitude"]
    longitude = location["longitude"]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code",
        },
    )

    if weather_response.status_code != 200:
        return f"Error: weather service returned status {weather_response.status_code}."

    current = weather_response.json()["current"]

    temp = current["temperature_2m"]
    wind = current["wind_speed_10m"]
    humidity = current["relative_humidity_2m"]
    code = current["weather_code"]
    condition = CODE_LOOKUP.get(code, "Unknown")

    return (
        f"Weather in {city}: {condition}, "
        f"{temp}°C, humidity {humidity}%, wind {wind} km/h."
    )


if __name__ == "__main__":
    print(get_weather("London"))
    print(get_weather("Lahore"))
    print(get_weather("neverlandpirates"))