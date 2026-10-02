"""Folium map: launch sites, per-launch outcome markers, proximity lines. Output: launch_map.html"""
import folium, pandas as pd
from folium.plugins import MarkerCluster, MousePosition
from math import radians, sin, cos, asin, sqrt
d = pd.read_csv("data/launches.csv")
S = {"CCAFS LC-40": (28.5623, -80.5774), "CCAFS SLC-40": (28.5632, -80.5772), "KSC LC-39A": (28.5730, -80.6469), "VAFB SLC-4E": (34.6321, -120.6107)}
m = folium.Map(location=[29.5, -90], zoom_start=4); cl = MarkerCluster().add_to(m); MousePosition().add_to(m)
for s, (la, lo) in S.items():
    folium.Circle((la, lo), 1000, color="#d35400", fill=True).add_child(folium.Popup(s)).add_to(m)
    for _, r in d[d.LaunchSite == s].iterrows():
        folium.Marker((la, lo), icon=folium.Icon(color="green" if r.Class else "red")).add_to(cl)
def km(a, b):
    la1, lo1, la2, lo2 = map(radians, (*a, *b)); h = sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2
    return 6371 * 2 * asin(sqrt(h))
coast = (28.5632, -80.5680)                      # nearest coastline point (read via MousePosition)
folium.PolyLine([S["KSC LC-39A"], coast]).add_child(folium.Popup(f"{km(S['KSC LC-39A'], coast):.2f} km to coast")).add_to(m)
m.save("launch_map.html")
