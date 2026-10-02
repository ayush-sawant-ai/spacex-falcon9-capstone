"""Scrape Wikipedia 'List of Falcon 9 and Falcon Heavy launches' tables -> data/falcon9_scraped.csv"""
import requests, pandas as pd
from bs4 import BeautifulSoup
URL = "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches"
soup = BeautifulSoup(requests.get(URL).text, "html.parser")
rows = []
for t in soup.find_all("table", class_="wikitable plainrowheaders collapsible"):
    for tr in t.find_all("tr")[1:]:
        c = [x.get_text(" ", strip=True) for x in tr.find_all(["th", "td"])]
        if len(c) >= 9: rows.append(c[:9])
cols = ["FlightNo","DateTime","Version","LaunchSite","Payload","PayloadMass","Orbit","Customer","Outcome"]
pd.DataFrame(rows, columns=cols).to_csv("data/falcon9_scraped.csv", index=False)
