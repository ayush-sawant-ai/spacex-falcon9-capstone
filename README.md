# SpaceX Falcon 9 First-Stage Landing Prediction (Data Science Capstone)

Predicts whether a Falcon 9 first stage will land successfully (cost: ~$62M vs ~$165M for competitors).

## Pipeline
1. `src/collect_spacex_api.py` – REST API collection (v4)
2. `src/scrape_wikipedia.py` – Wikipedia HTML table scraping (BeautifulSoup)
3. `src/analysis_and_report.py` – wrangling, EDA, SQL (sqlite), Folium/Dash views, ML models, builds the PDF report

## Run
    pip install -r requirements.txt
    python src/collect_spacex_api.py && python src/scrape_wikipedia.py
    python src/analysis_and_report.py

> NOTE: `analysis_and_report.py` ships with a simulated dataset (`data/launches.csv`) shaped like the real
> one so the report builds offline. Swap in the collected CSV for real figures.
