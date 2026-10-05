# eThekwini Election Observatory

A local Streamlit dashboard for the supplied eThekwini proportional-representation election exports. The notebook and CSV files remain unchanged.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Use Python 3.11 or newer. The three `ETH_PR_YYYY.csv` inputs must remain beside `app.py`; paths are resolved from the app location, not the terminal's working directory. All analysis runs locally with no account, API key, or external data service.

## Pages

- **Metro overview:** historical results beside hatched 2026 prediction bars; turnout, registration, vote volumes, historical trends and comparison downloads.
- **Ward explorer:** searchable selector for all 111 reference wards, party shares, historical trends, projected PR leaders, close-lead flags and an all-ward table.
- **Model performance:** computed 2021 holdout metrics against a persistence baseline, party MAE/RMSE, observed/predicted plot, leader confusion matrix, precision/recall/F1 and turnout errors.
- **Limitations & uncertainty:** model rules, uncertainty interpretation, missing newer results, boundary caveats, source coverage and district-level anomalies.

Sidebar party filters affect exploration charts, comparison tables and downloads. They never renormalize shares or alter totals, leaders or evaluation metrics. Empty selections produce explanatory messages. Historical year and ward selections persist between views. Scenario controls adjust registration growth, turnout and spoilt-ballot assumptions; the performance page always evaluates the fixed model.

## Method and interpretation

The notebook's swing estimator is reproduced as `latest ward share + 0.5 × local swing + 0.5 × metro swing`, clipped and normalized over all five categories. Turnout is `0.7 × latest + 0.3 × previous`, clipped to 0–100%. The 2026 defaults assume 2% registration growth and 1.5% spoilt ballots. Metro forecasts aggregate ward vote volumes, rather than averaging shares.

The fixed categories are ANC, DA, EFF, IFP and **all remaining parties combined**. Unlike the notebook's year-dependent threshold and subsequent five-category subset, this preserves every PR vote and a consistent denominator. DA's bilingual historical name is reconciled. Registration and spoilt votes are counted once per voting district, not per party record. The mixed 2016 file is filtered to PR before aggregation.

Historical voting districts map to 2021 wards by ID. Unmapped districts remain in metro history and are excluded, with coverage reported, from ward history. This is a retrospective reference geography, not a verified spatial crosswalk or verified 2026 geography. The 2021 holdout uses only 2011/2016 vote and turnout features, but the 2021 geography is retrospective. The 110 wards with observations in all three cycles enter evaluation. Missing earlier observations in forecasting use persistence.

Sensitivity bars show **± party-specific ward MAE from the 2021 backtest**, not confidence intervals. Applying those errors to metro shares is illustrative only. No claimed coverage probability, new-party support, seats, elected ward candidates or governing outcome is estimated. `Other` is never declared a party winner; a tie or an aggregate Other share above the named leader is marked Unresolved. Inputs after 2021 are absent, so outputs are experimental historical-data scenarios.

Further corrections from the notebook: consistent turnout logic in backtest and forecast, no arbitrary 20% turnout floor, no hardcoded performance metrics, and no unverified election-date claim. Historical source anomalies are retained and displayed.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Checks cover source totals, ballot filtering, registration deduplication, geography reconciliation, forecast bounds and vote conservation, future-value leakage, missing-history fallback, error calculations and dashboard interactions.

Core files: `app.py` (interface), `election_data.py` (ingestion/model/backtest), `.streamlit/config.toml` (theme). Source-file modification times invalidate the data cache. The interface uses Streamlit's [application testing API](https://docs.streamlit.io/develop/api-reference/app-testing) for interaction checks.
