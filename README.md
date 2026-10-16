# QueueSight – hospital delay early-warning (Datathon 2026, Health: The Forgotten Patient)

Synthetic data only – not for clinical use.

## Run it (from this folder, after `pip install -r requirements.txt`)
```
python src/01_explore_data.py        # checks the data, saves charts to out/figures/
python src/train_queuesight.py data  # trains models, writes out/ (metrics, alert queue, models)
streamlit run app/app.py             # opens the dashboard in your browser
```

## Folder map
- `data/`  the 4 dataset CSVs
- `src/`   `01_explore_data.py` (look at the data), `train_queuesight.py` (the ML pipeline)
- `app/`   `app.py` (the Streamlit dashboard)
- `out/`   generated results (metrics.json, priority_queue.csv, saved models, charts)
