"""Wrangling, EDA, SQL, map/dashboard views, ML and PDF report (offline, simulated data)."""
import numpy as np, pandas as pd, sqlite3, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

GH = "https://github.com/YOUR-USERNAME/spacex-falcon9-capstone"   # <- replace after pushing
NAVY, BLUE, ORG, GRY = "#0b2545", "#1f77b4", "#f28e2b", "#555555"
plt.rcParams["text.parse_math"] = False
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

# ---------- simulated dataset shaped like the real one (90 Falcon 9 launches) ----------
n = 90
sites = {"CCAFS LC-40": (28.5623, -80.5774), "CCAFS SLC-40": (28.5632, -80.5772),
         "KSC LC-39A": (28.5730, -80.6469), "VAFB SLC-4E": (34.6321, -120.6107)}
d = pd.DataFrame({"FlightNumber": np.arange(1, n + 1)})
d["Date"] = pd.date_range("2010-06-04", "2020-11-30", periods=n)
d["Year"] = d.Date.dt.year
d["LaunchSite"] = [rng.choice(list(sites), p=[.38, .27, .22, .13] if i > 20 else [.7, .1, .05, .15]) for i in range(n)]
d["Orbit"] = rng.choice(["LEO","ISS","VLEO","PO","GTO","SSO","MEO","ES-L1","HEO","GEO"], n,
                         p=[.1,.23,.15,.1,.2,.1,.04,.03,.02,.03])
d["PayloadMass"] = np.where(d.Orbit.isin(["GTO","GEO","HEO"]), rng.normal(5500, 1500, n),
                   np.where(d.Orbit == "VLEO", rng.normal(15000, 800, n), rng.normal(3500, 2200, n))).clip(300, 16000).round()
d["Reused"] = (d.FlightNumber > 40) & (rng.random(n) < .5)
d["GridFins"] = d.FlightNumber > 12
d["Legs"] = d.FlightNumber > 12
p = 1 / (1 + np.exp(-(0.075 * (d.FlightNumber - 38)))) * np.where(d.Orbit.isin(["GEO","HEO","ES-L1"]), .9, 1)
d["Class"] = (rng.random(n) < p).astype(int)
d.loc[d.FlightNumber.isin([1, 2, 3, 4]), "Class"] = 0
d.to_csv("data/launches.csv", index=False)

# ---------- SQL (sqlite) ----------
con = sqlite3.connect(":memory:"); d.assign(Date=d.Date.astype(str)).to_sql("SPACEXTBL", con, index=False)
Q = {
 "Unique launch sites": "SELECT DISTINCT LaunchSite FROM SPACEXTBL",
 "5 records, site starts 'CCA'": "SELECT FlightNumber, LaunchSite, Orbit, PayloadMass FROM SPACEXTBL WHERE LaunchSite LIKE 'CCA%' LIMIT 5",
 "Total payload (ISS orbit, kg)": "SELECT SUM(PayloadMass) FROM SPACEXTBL WHERE Orbit='ISS'",
 "Avg payload, reused boosters (kg)": "SELECT ROUND(AVG(PayloadMass)) FROM SPACEXTBL WHERE Reused=1",
 "First successful landing date": "SELECT MIN(Date) FROM SPACEXTBL WHERE Class=1",
 "Heaviest payload landed (kg)": "SELECT MAX(PayloadMass) FROM SPACEXTBL WHERE Class=1",
 "Success vs failure count": "SELECT Class, COUNT(*) FROM SPACEXTBL GROUP BY Class",
 "Landings per year (ranked)": "SELECT Year, SUM(Class) AS landings FROM SPACEXTBL GROUP BY Year ORDER BY landings DESC",
 "Success rate by site (ranked)": "SELECT LaunchSite, ROUND(AVG(Class),2) AS rate FROM SPACEXTBL GROUP BY LaunchSite ORDER BY rate DESC",
}
R = {k: pd.read_sql(v, con) for k, v in Q.items()}

# ---------- ML ----------
orb = pd.get_dummies(d[["Orbit", "LaunchSite"]]).astype(float)
X = pd.concat([d[["FlightNumber", "PayloadMass"]], d[["Reused", "GridFins", "Legs"]].astype(float), orb], axis=1)
Xs = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(Xs, d.Class, test_size=.2, random_state=2)
models = {
 "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [.01, .1, 1], "solver": ["lbfgs"]}),
 "SVM": (SVC(), {"kernel": ["rbf", "linear"], "C": [.1, 1, 10]}),
 "Decision Tree": (DecisionTreeClassifier(random_state=2), {"max_depth": [2, 4, 6, 8], "criterion": ["gini", "entropy"]}),
 "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9], "p": [1, 2]}),
}
res, fit = {}, {}
for k, (m, g) in models.items():
    gs = GridSearchCV(m, g, cv=10).fit(Xtr, ytr); fit[k] = gs
    res[k] = (gs.best_score_, gs.score(Xte, yte))
best = max(res, key=lambda k: res[k][1]); pred = fit[best].predict(Xte); cmx = confusion_matrix(yte, pred)

# ---------- PDF helpers ----------
pdf = PdfPages("Data_Science_Capstone_Project_Report.pdf"); page = [0]
def slide(title, gh=False):
    fig = plt.figure(figsize=(13.33, 7.5)); page[0] += 1
    fig.patches.append(plt.Rectangle((0, .90), 1, .10, transform=fig.transFigure, color=NAVY))
    fig.text(.04, .935, title, color="white", fontsize=24, weight="bold", va="center")
    fig.text(.96, .02, str(page[0]), color=GRY, ha="right", fontsize=10)
    if gh: fig.text(.04, .02, "GitHub: " + GH, color=BLUE, fontsize=11)
    return fig
def bullets(fig, items, x=.05, y=.82, size=16, gap=.075, w=None):
    for t in items:
        fig.text(x, y, "•  " + t, fontsize=size, va="top", color="#222", wrap=True); y -= gap
def box(ax, x, y, w, h, t, c=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.01", fc=c, ec="none"))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=12, weight="bold")
def flow(fig, steps, y=.5):
    ax = fig.add_axes([.03, .15, .94, .6]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    w = .92 / len(steps) - .03
    for i, s in enumerate(steps):
        x = .02 + i * (w + .03 + 0)
        box(ax, x, .4, w, .25, s, BLUE if i % 2 == 0 else ORG)
        if i < len(steps) - 1: ax.annotate("", (x + w + .03, .525), (x + w, .525), arrowprops=dict(arrowstyle="->", lw=2))
def plain(ax=None):
    ax.spines[["top", "right"]].set_visible(False)
def table(fig, df, rect, title=None, fs=11):
    ax = fig.add_axes(rect); ax.axis("off")
    if title: ax.set_title(title, loc="left", fontsize=13, weight="bold", color=NAVY)
    t = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns), loc="upper left", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white")
def save(fig): pdf.savefig(fig); plt.close(fig)

sr = d.Class.mean(); early = d[d.FlightNumber <= 30].Class.mean(); late = d[d.FlightNumber > 60].Class.mean()

# 1 Title
fig = plt.figure(figsize=(13.33, 7.5)); page[0] += 1
fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=NAVY))
fig.text(.5, .62, "Data Science Capstone Project Report", color="white", fontsize=36, weight="bold", ha="center")
fig.text(.5, .50, "Predicting SpaceX Falcon 9 First-Stage Landing Success", color=ORG, fontsize=22, ha="center")
fig.text(.5, .36, "Your Name  |  October 2026", color="white", fontsize=16, ha="center")
fig.text(.5, .28, "GitHub: " + GH, color="#9cc9ff", fontsize=14, ha="center")
fig.text(.5, .06, "Figures use a simulated dataset shaped like the real Falcon 9 data; rerun the repo on API data for real results.",
         color="#9cc9ff", fontsize=10, ha="center"); save(fig)

# 2 Exec summary
fig = slide("Executive Summary"); bullets(fig, [
 "Goal: predict whether a Falcon 9 first stage lands, which drives launch cost (~$62M vs ~$165M).",
 "Methods: SpaceX REST API + Wikipedia scraping, pandas wrangling, EDA (matplotlib), SQL (sqlite),",
 "      Folium maps, Plotly Dash dashboard, and ML (LogReg, SVM, Decision Tree, KNN with 10-fold GridSearchCV).",
 f"Key results: landing success rises from {early:.0%} (flights 1-30) to {late:.0%} (flights 61+).",
 "Flight number (accumulated experience) is the clearest driver of success in this dataset; orbit, payload and site are secondary.",
 f"Best model on the test set: {best} at {res[best][1]:.0%} accuracy (all four models within a narrow band).",
 "Takeaway: a competitor can estimate Falcon 9 launch cost by forecasting booster reuse probability."], size=15, gap=.085)
save(fig)

# 3 Intro
fig = slide("Introduction"); bullets(fig, [
 "Background: Falcon 9 launches cost ~$62M vs ~$165M+ for others, because Stage 1 is reusable.",
 "If we can predict whether Stage 1 will land, we can predict launch cost.",
 "Problem statement: given launch features (payload, orbit, site, flight number, reuse, grid fins, legs),",
 "      classify each launch as landing success (1) or failure (0).",
 "Questions: Which factors drive success? Do sites differ? How did success improve over time?",
 "Audience: a new rocket company bidding against SpaceX."], size=16, gap=.09); save(fig)

# 4 API
fig = slide("Data Collection - SpaceX API", gh=True)
flow(fig, ["GET /v4/launches/past", "Filter rocket =\nFalcon 9", "GET /payloads,\n/launchpads, /cores", "json_normalize\n+ merge", "data/\nspacex_api_raw.csv"])
fig.text(.05, .30, "Key phrases: requests.get() | response.json() | pd.json_normalize() | ID look-ups cached | drop Falcon 1 | save CSV", fontsize=13)
save(fig)
# 5 Scraping
fig = slide("Data Collection - Web Scraping", gh=True)
flow(fig, ["requests.get\nWikipedia page", "BeautifulSoup\nparse HTML", "Find wikitable\nlaunch tables", "Extract rows\n-> DataFrame", "data/falcon9_\nscraped.csv"])
fig.text(.05, .30, "Key phrases: soup.find_all('table') | row cells -> text | booster landing outcome | payload mass | orbit | customer", fontsize=13)
save(fig)
# 6 Wrangling
fig = slide("Data Wrangling Methodology", gh=True); bullets(fig, [
 "Merged API and scraped tables on flight number; kept Falcon 9 launches only.",
 "Imputed missing PayloadMass with the column mean; kept LandingPad nulls (= no pad used).",
 "Counted launches per site and orbit, and landing outcomes (True/False Ocean, RTLS, ASDS, None).",
 "Created binary label Class: 1 = successful landing, 0 = otherwise.",
 f"Result: {n} rows, overall success rate {sr:.0%}.",
 "One-hot encoded Orbit and LaunchSite, standardised features before modelling."], size=16, gap=.09); save(fig)
# 7 EDA viz summary
fig = slide("EDA with Data Visualization", gh=True); bullets(fig, [
 "Scatter: FlightNumber vs LaunchSite (hue = outcome) - does experience help at each site?",
 "Scatter: PayloadMass vs LaunchSite - does payload weight limit success?",
 "Bar: success rate by orbit - which missions are safest?",
 "Scatter: FlightNumber vs Orbit - orbit-specific learning curve.",
 "Line: yearly success-rate trend - overall improvement.",
 "Purpose: spot patterns that guide feature selection for the classifier."], size=16, gap=.09); save(fig)
# 8 SQL summary
fig = slide("EDA with SQL", gh=True); bullets(fig, [
 "Loaded data into SQLite (SPACEXTBL) and ran 9 queries.",
 "DISTINCT launch sites; LIKE 'CCA%' filter with LIMIT 5.",
 "SUM / AVG of payload by orbit and for reused boosters.",
 "MIN(date) of first successful landing; MAX payload landed.",
 "GROUP BY outcome counts, landings per year, success rate per site.",
 "ORDER BY ... DESC to rank years and sites."], size=16, gap=.09); save(fig)
# 9 Interactive summary
fig = slide("Interactive Visual Analytics", gh=True); bullets(fig, [
 "Folium: markers for 4 launch sites, green/red outcome markers per launch (MarkerCluster).",
 "Folium: MousePosition + PolyLine to measure distance to coast, railway, highway and city.",
 "Plotly Dash: dropdown for site + RangeSlider for payload.",
 "Dash: success pie chart (all sites or one site) and payload-vs-outcome scatter by booster version.",
 f"Dash app: {GH}/blob/main/src/dash_app.py   |   Map: {GH}/blob/main/src/folium_map.py"], size=15, gap=.09); save(fig)

# 10 Flight vs site
cm = {0: "#d62728", 1: "#2ca02c"}
fig = slide("EDA: Flight Number vs Launch Site"); ax = fig.add_axes([.14, .1, .81, .72])
for s_i, s in enumerate(sites):
    for c in (0, 1):
        q = d[(d.LaunchSite == s) & (d.Class == c)]; ax.scatter(q.FlightNumber, [s_i] * len(q) + rng.normal(0, .06, len(q)), c=cm[c], alpha=.7, s=60, label=f"Class {c}" if s_i == 0 else None)
ax.set_yticks(range(4)); ax.set_yticklabels(sites); ax.set_xlabel("Flight Number"); ax.legend(); plain(ax)
fig.text(.07, .84, "Success clusters at higher flight numbers at every site: experience improves outcomes.", fontsize=13, color=GRY); save(fig)
# 11 Payload vs site
fig = slide("EDA: Payload vs Launch Site"); ax = fig.add_axes([.14, .1, .81, .72])
for s_i, s in enumerate(sites):
    for c in (0, 1):
        q = d[(d.LaunchSite == s) & (d.Class == c)]; ax.scatter(q.PayloadMass, [s_i] * len(q) + rng.normal(0, .06, len(q)), c=cm[c], alpha=.7, s=60)
ax.set_yticks(range(4)); ax.set_yticklabels(sites); ax.set_xlabel("Payload Mass (kg)"); plain(ax)
fig.text(.07, .84, f"Payloads cluster at 2-7 t; a separate group above 14 t (VLEO). Heavy-payload success rate: {d[d.PayloadMass>14000].Class.mean():.0%}.", fontsize=13, color=GRY); save(fig)
# 12 Orbit bar
fig = slide("EDA: Success Rate by Orbit"); ax = fig.add_axes([.07, .12, .88, .68])
o = d.groupby("Orbit").Class.mean().sort_values(ascending=False); ax.bar(o.index, o.values, color=BLUE); ax.set_ylabel("Success rate"); plain(ax)
fig.text(.07, .84, f"Highest: {o.index[0]} ({o.iloc[0]:.0%}); lowest: {o.index[-1]} ({o.iloc[-1]:.0%}). Small orbit groups (n<=5) make these rates noisy.", fontsize=13, color=GRY); save(fig)
# 13 Flight vs orbit + yearly
fig = slide("EDA: Flight Number vs Orbit and Yearly Trend")
ax = fig.add_axes([.06, .12, .42, .68]); ol = list(d.Orbit.unique())
for c in (0, 1):
    q = d[d.Class == c]; ax.scatter(q.FlightNumber, [ol.index(x) for x in q.Orbit], c=cm[c], s=50, alpha=.7)
ax.set_yticks(range(len(ol))); ax.set_yticklabels(ol); ax.set_xlabel("Flight Number"); plain(ax)
ax = fig.add_axes([.56, .12, .4, .68]); y = d.groupby("Year").Class.mean(); ax.plot(y.index, y.values, marker="o", color=ORG, lw=3)
ax.set_xlabel("Year"); ax.set_ylabel("Success rate"); plain(ax); ax.set_title("Yearly success-rate trend")
save(fig)

# 14 SQL results 1
fig = slide("EDA with SQL: Sites, Payloads, Dates")
table(fig, R["Unique launch sites"], [.04, .45, .3, .35], "Unique launch sites")
table(fig, R["5 records, site starts 'CCA'"], [.40, .45, .56, .35], "5 records where site starts with 'CCA'", 10)
sc = pd.DataFrame({"Query": ["Total payload, ISS (kg)", "Avg payload, reused (kg)", "First successful landing", "Heaviest payload landed (kg)"],
                   "Result": [R["Total payload (ISS orbit, kg)"].iloc[0, 0], R["Avg payload, reused boosters (kg)"].iloc[0, 0],
                              str(R["First successful landing date"].iloc[0, 0])[:10], R["Heaviest payload landed (kg)"].iloc[0, 0]]})
table(fig, sc, [.04, .1, .5, .3], "Aggregate queries", 11); save(fig)
# 15 SQL results 2
fig = slide("EDA with SQL: Success Rates, Rankings, Time")
t = R["Success vs failure count"].rename(columns={"COUNT(*)": "Count"}); table(fig, t, [.04, .5, .25, .3], "Outcome counts (Class)")
table(fig, R["Success rate by site (ranked)"], [.04, .15, .4, .3], "Success rate by site (ranked)")
table(fig, R["Landings per year (ranked)"].head(7), [.55, .3, .4, .5], "Top years by landings", 11); save(fig)

# 16 Folium markers
fig = slide("Folium Map: Launch Sites and Launch Records", gh=True)
ax = fig.add_axes([.05, .1, .55, .75]); ax.set_facecolor("#dbe9f6")
for s, (la, lo) in sites.items():
    ax.scatter(lo, la, s=250, c=NAVY, marker="^", zorder=3); off = {"VAFB SLC-4E": (12, 0), "CCAFS LC-40": (-130, -30), "CCAFS SLC-40": (-130, 0), "KSC LC-39A": (-130, 30)}[s]
    ax.annotate(s, (lo, la), textcoords="offset points", xytext=off, fontsize=10, arrowprops=dict(arrowstyle="-", color=GRY))
ax.set_xlim(-125, -70); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Launch sites (schematic of Folium markers)"); plain(ax)
bullets(fig, ["3 of 4 sites are in Florida;", "1 (VAFB) is in California.", "All sites sit close to the coast", "(launches go over ocean).", "",
              "Cluster markers: green = success,", "red = failure; count per site", "shown on cluster badge.",
              f"Best site by rate: {R['Success rate by site (ranked)'].iloc[0, 0]}"], x=.65, y=.8, size=14, gap=.06); save(fig)
# 17 Proximity
fig = slide("Folium Map: Proximity Analysis", gh=True)
prox = pd.DataFrame({"Feature": ["Coastline", "Railway", "Highway", "Nearest city (Titusville)"], "Distance (km)": [0.9, 1.3, 0.6, 23.0]})
table(fig, prox, [.05, .45, .45, .35], "KSC LC-39A (approx., measured with MousePosition + PolyLine)", 13)
bullets(fig, ["Near coastline: yes - rockets fly over water to limit risk.", "Near railway/highway: yes - heavy equipment transport.",
              "Far from cities: yes - safety buffer from populated areas."], x=.05, y=.35, size=15, gap=.08)
fig.text(.05, .08, "Distances are illustrative placeholders; the Folium script computes them with the haversine formula.", fontsize=10, color=GRY); save(fig)

# 18 Dash pie
fig = slide("Plotly Dash: Success Pie Charts", gh=True)
ax = fig.add_axes([.04, .1, .42, .7]); s_all = d.groupby("LaunchSite").Class.sum(); ax.pie(s_all, labels=s_all.index, autopct="%1.0f%%"); ax.set_title("All sites: share of total successes")
ax = fig.add_axes([.54, .1, .42, .7]); bs = d.groupby("LaunchSite").Class.mean().idxmax(); q = d[d.LaunchSite == bs].Class.value_counts()
ax.pie(q, labels=["Success" if i else "Failure" for i in q.index], autopct="%1.0f%%", colors=["#2ca02c", "#d62728"][:len(q)] if q.index[0] == 1 else ["#d62728", "#2ca02c"]); ax.set_title(f"Highest success-ratio site: {bs}")
save(fig)
# 19 Dash scatter
fig = slide("Plotly Dash: Payload vs Launch Outcome", gh=True); ax = fig.add_axes([.07, .1, .88, .7])
for c in (0, 1):
    q = d[d.Class == c]; ax.scatter(q.PayloadMass, q.Class + rng.normal(0, .03, len(q)), c=cm[c], s=60, alpha=.7)
ax.axvspan(2000, 6000, alpha=.1, color=ORG); ax.set_yticks([0, 1]); ax.set_yticklabels(["Failure", "Success"]); ax.set_xlabel("Payload Mass (kg)"); plain(ax)
sel = d[(d.PayloadMass >= 2000) & (d.PayloadMass <= 6000)].Class.mean(); allr = d.Class.mean()
fig.text(.07, .84, f"RangeSlider 2,000-6,000 kg (shaded): success {sel:.0%} vs {allr:.0%} overall - compare with the full range via the slider.", fontsize=13, color=GRY); save(fig)

# 20 Models
fig = slide("Predictive Analysis: Model Comparison", gh=True)
ax = fig.add_axes([.07, .12, .5, .68]); k = list(res); ax.bar(k, [res[x][0] for x in k], .38, label="CV accuracy", color=BLUE)
ax.bar(np.arange(4) + .38, [res[x][1] for x in k], .38, label="Test accuracy", color=ORG); ax.set_xticks(np.arange(4) + .19); ax.set_xticklabels(k, fontsize=9); ax.set_ylim(0, 1); ax.legend(loc="lower right"); plain(ax)
tb = pd.DataFrame({"Model": k, "CV": [f"{res[x][0]:.2f}" for x in k], "Test": [f"{res[x][1]:.2f}" for x in k]})
table(fig, tb, [.62, .35, .35, .4], "Results (10-fold GridSearchCV)", 12)
fig.text(.62, .25, f"Best params ({best}):\n{fit[best].best_params_}", fontsize=11); save(fig)
# 21 Confusion matrix
fig = slide("Predictive Analysis: Confusion Matrix and Best Model"); ax = fig.add_axes([.06, .12, .42, .7])
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=["Did not land", "Landed"]).plot(ax=ax, cmap="Blues", colorbar=False)
bullets(fig, [f"Best model: {best} (test acc {res[best][1]:.0%}).", "Test set is only 18 launches, so differences", "between models are within noise.",
              f"Errors: {cmx[0,1]} false positive(s), {cmx[1,0]} false negative(s).", "Flight number is the main signal in the data.", "",
              "Next step: more data + cross-validated", "model comparison with confidence intervals."], x=.55, y=.78, size=14, gap=.07); save(fig)
# 22 Conclusion
fig = slide("Conclusion and Insights"); bullets(fig, [
 f"Landing success grew from {early:.0%} to {late:.0%} as flights accumulated: experience drives reliability.",
 "Orbit and payload effects are visible but noisy at this sample size (see orbit bar chart).",
 f"{best} reaches {res[best][1]:.0%} test accuracy; all four models give a similar level of accuracy.",
 "Creative insight: pair the classifier with cost: expected cost = $165M - P(land) x ~$100M saving,",
 "      so a rival can price bids per mission profile.",
 "Insight: a 'learning-curve' feature (flights flown by the core version) would likely beat raw flight number.",
 "Future work: add weather, booster block and landing-pad data; try gradient boosting; deploy the Dash app."], size=14, gap=.085); save(fig)
pdf.close(); print("done", res, best)
