# SupeRR Selector 3.0 — Live Deployment Copy

A Streamlit dashboard for the SupeRR Selector Season 2 pressure-performance analysis. This is a deployment-ready copy; the original SupeRR Selector 2.0 and 3.0 backup packages are not modified.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

The supplied hackathon datasets are bundled under `data/` so judges do not need to upload files. The large ball-by-ball file is stored as gzip (`ball_by_ball.csv.gz`) to keep the repository compact; pandas reads it directly. Supporting files include matches, competitions, player mapping, and playing XI data.

The dashboard pools IPL and T20I into one batter ranking and one bowler ranking. It includes player comparison, pressure distributions, downloadable scorecards, and chronological model validation.

## Model settings

Defaults: high-pressure threshold 55; minimum 100 pressure balls; minimum 10 pressure matches; minimum 250 total role-specific balls; rating weights 40% contextual impact, 35% personal uplift, 25% consistency.

## Important interpretation notes

This is an analytical prototype. The pressure components, wicket run-equivalent, and score weights are transparent modelling assumptions, not organizer-prescribed rules. Batting order is an imperfect responsibility proxy. First-innings par run rate is estimated from the supplied dataset by format, phase, and wicket band; it is not venue- or pitch-adjusted. The playing-XI file is loaded for supporting context, but its leadership flag does not currently alter the ranking score. Rankings should be presented as model-based estimates with these limitations, not as definitive measures of player quality.

## Deployment

Deploy `app.py` from this repository on Streamlit Community Cloud. Keep `requirements.txt` at the repository root. Once deployed, check the app's Sharing settings and make the app public so hackathon judges can access it.
