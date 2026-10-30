# UAC Program – Predictive Forecasting of Care Load & Placement Demand

## Run the dashboard
    pip install -r requirements.txt
    streamlit run app.py

## Reproduce the analysis (order matters; ~25 min on 2 CPU cores, mostly step 2)
    python src/prep.py       # clean + daily reindex  -> results/data_clean.csv
    python src/eda.py        # EDA figures / stats
    python src/forecast.py   # walk-forward backtests + final forecasts
    python src/metrics.py    # MAE/RMSE/MAPE, intervals, KPIs
    python src/figs2.py      # result figures, feature importance

Raw data: data_raw.csv (HHS_Unaccompanied_Alien_Children_Program.csv).
Main test: 1 Apr–21 Dec 2025. Stress test: 15 Dec 2024–31 Mar 2025.
