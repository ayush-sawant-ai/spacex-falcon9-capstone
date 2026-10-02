"""Plotly Dash dashboard: site dropdown, success pie, payload slider, payload-vs-outcome scatter."""
import pandas as pd, plotly.express as px
from dash import Dash, dcc, html, Input, Output
d = pd.read_csv("data/launches.csv"); app = Dash(__name__)
app.layout = html.Div([
    html.H1("SpaceX Launch Records"),
    dcc.Dropdown(["ALL"] + sorted(d.LaunchSite.unique()), "ALL", id="site"),
    dcc.Graph(id="pie"),
    dcc.RangeSlider(0, 16000, 1000, value=[0, 16000], id="payload"),
    dcc.Graph(id="scatter")])
@app.callback(Output("pie", "figure"), Input("site", "value"))
def pie(site):
    if site == "ALL": return px.pie(d, values="Class", names="LaunchSite", title="Successes by site")
    return px.pie(d[d.LaunchSite == site], names="Class", title=f"Outcomes at {site}")
@app.callback(Output("scatter", "figure"), Input("site", "value"), Input("payload", "value"))
def sc(site, pl):
    q = d if site == "ALL" else d[d.LaunchSite == site]
    q = q[q.PayloadMass.between(*pl)]
    return px.scatter(q, x="PayloadMass", y="Class", color="Orbit", title="Payload vs outcome")
if __name__ == "__main__": app.run(debug=True)
