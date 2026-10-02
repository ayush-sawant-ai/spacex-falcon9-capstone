"""Collect Falcon 9 launches from the SpaceX REST API (v4) -> data/spacex_api_raw.csv"""
import requests, pandas as pd
B = "https://api.spacexdata.com/v4"
df = pd.json_normalize(requests.get(f"{B}/launches/past").json())
df = df[df.rocket == "5e9d0d95eda69973a809d1ec"].reset_index(drop=True)   # Falcon 9 only
get = lambda path, i: requests.get(f"{B}/{path}/{i}").json() if i else {}
df["PayloadMass"] = [get("payloads", p[0]).get("mass_kg") for p in df.payloads]
df["LaunchSite"] = [get("launchpads", i).get("name") for i in df.launchpad]
core = pd.json_normalize(df.cores.str[0])[["landing_success","landing_type","reused","gridfins","legs"]]
pd.concat([df[["flight_number","name","date_utc","PayloadMass","LaunchSite"]], core], axis=1).to_csv("data/spacex_api_raw.csv", index=False)
