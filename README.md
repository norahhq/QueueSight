#  QueueSight

**A hospital delay early-warning tool that answers one question: *what is likely to get delayed next, why, and what should operations examine first?***


> ⚠️ **Disclaimer:** All data is synthetic. QueueSight is an operations-analytics prototype. It is **not** a clinical tool and must not be used for diagnosis or treatment decisions.

---

## Table of contents
1. [The problem](#the-problem)
2. [The solution](#the-solution)
3. [Key findings](#key-findings)
4. [Model and results](#model-and-results)
5. [How it works](#how-it-works)
6. [Quick start](#quick-start)
7. [Project structure](#project-structure)
8. [Dashboard guide](#dashboard-guide)
9. [Methodology notes](#methodology-notes)
10. [Limitations](#limitations)
11. [Roadmap](#roadmap)
12. [Deploying the dashboard](#deploying-the-dashboard)

---

## The problem
A patient's hospital journey is connected: arrival → triage → consultation → diagnostic tests → treatment → discharge. When each stage is managed separately, delays and capacity pressure are hard to spot early. Operations managers, department coordinators and service administrators need a clear view of **where patients are waiting, which services are under pressure, and where to intervene.**

A plain "average wait" view is misleading: low-acuity patients *should* wait longer than critical ones. A fair view must compare each service with what is normal **for its acuity mix.**

## The solution
QueueSight has four parts:

| Module | What it does |
|---|---|
| **Patient-flow dashboard** | Department wait times, triage breakdown, safety KPIs |
| **Wait and delay predictor** | Expected wait, a 90% range, and breach risk for any department × triage × time |
| **Triage-adjusted alerts** | Flags services running *worse than expected for their acuity* |
| **Investigate-first queue** | Ranks departments by **impact × confidence**, with an explanation panel for each alert |

It also includes a **diagnostic-test turnaround monitor** that catches "priority inversion" (Urgent tests that are no faster than Routine ones).

## Key findings
Computed from 8,000 encounters, 2,000 patients, 12 departments and 5,000 tests:

- **Waits follow triage, as expected:** Critical 17 min → High 27 → Medium 49 → Low 72. Raw wait is therefore a poor alert signal.
- **Same acuity, different wait:** High-triage patients wait about 36 min in Laboratory vs about 20 min in Cardiology and Admissions.
- **37%** of Critical/High patients wait more than 30 minutes.
- **Departments differ by up to 36%:** Pediatrics and Laboratory (61 min) vs Pharmacy and Admissions (45 min).
- **Hour, weekday and daily load barely affect wait**, so the delays are structural, not a rush-hour effect.
- **Tests:** 14% take longer than 2 hours; Urgent tests (69 min) are no faster than Routine (70 min).

![Wait by triage](out/figures/wait_by_triage.png)
![High-triage wait by department](out/figures/high_triage_by_department.png)

## Model and results
Models are trained **from scratch** on the provided data. No external API or pretrained model is used.

| Component | Method | Purpose |
|---|---|---|
| Wait predictor | Gradient boosting regressor (scikit-learn `HistGradientBoostingRegressor`) | Expected wait in minutes |
| Prediction interval | 5th/95th percentile of training residuals | 90% range around each prediction |
| Breach classifier | Gradient boosting classifier | Probability of exceeding the triage-based target |
| Alert scoring | Residual vs expected → z-score → confidence | Which departments are worse than normal |

**Evaluation uses a time-aware split** (train Jan–Aug, validate Sep–Oct, test Nov–Dec), never a random shuffle.

| Metric (Nov–Dec holdout) | Result |
|---|---|
| MAE, naive global-average baseline | 19.7 min |
| MAE, triage-average baseline | 14.3 min |
| MAE, department × triage average baseline | 13.5 min |
| **MAE, QueueSight model** | **13.6 min (31% lower than naive)** |
| R² | 0.51 |
| 90% interval coverage | 89.7% |
| Breach classifier AUC | 0.68 |

**Honest note:** the model performs on par with a strong department × triage average. Permutation importance shows only **triage level** and **department** carry real signal in this synthetic data; hour, age, visit type and rolling load add roughly nothing. The value QueueSight adds is calibrated uncertainty, breach risk, and the ranked alert queue built on top.

## How it works
```
 CSV files ─► join & validate ─► leak-free features ─► time-aware split ─► train models
                                                                              │
         dashboard (Streamlit) ◄── out/ (metrics, queue, models) ◄── alerts ◄─┘
```

**Alert logic (Investigate-first queue):**
1. Train on history up to 28 days before the end of the data.
2. For recent visits, compute `excess = actual wait − predicted wait`.
3. Per department: `impact = mean(excess) × patients`; `z = mean(excess) × √n / σ`; `confidence = Φ(z)`.
4. `priority = impact × confidence`, sorted highest first.
5. Each alert shows its main driver (the triage level contributing most excess).

## Quick start
**Requirements:** Python 3.10 or newer.

**Easiest:** double-click `run_windows.bat` (Windows) or run:
```bash
bash run_mac_linux.sh
```

**Manual:**
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/01_explore_data.py      # data checks + charts in out/figures/
python src/train_queuesight.py data   # trains models, writes out/
streamlit run app/app.py           # opens http://localhost:8501
```
Trained models are already included in `out/`. The app retrains automatically if they are missing or were created with a different library version.

## Project structure
```
queuesight_project/
├── app/
│   └── app.py                  # Streamlit dashboard (4 tabs)
├── src/
│   ├── 01_explore_data.py      # data-quality checks and charts
│   └── train_queuesight.py     # full ML pipeline: features → train → evaluate → alerts
├── data/                       # departments, patients, encounters, diagnostic_tests (CSV) + dataset README
├── out/                        # generated: metrics.json, priority_queue.csv, weekly_watchlist.csv, models, figures
├── docs/                       # screenshots for this README
├── requirements.txt
├── run_windows.bat
├── run_mac_linux.sh
└── README.md
```

## Dashboard guide
| Tab | What you can do |
|---|---|
| **Overview** | Filter by triage level; see visits, average wait, % of Critical/High waiting over 30 min, tests over 2 h, and wait by department/triage |
| **Predict a wait** | Choose department, triage, visit type and hour; get expected wait, 90% range and breach risk |
| **Investigate first** | Ranked alert table; choose a department to see why it was flagged and its excess wait by triage level |
| **Test monitor** | Turnaround by test type (average and P90), Urgent vs Routine comparison |

<!-- Add your screenshots to docs/ and uncomment:
![Overview](docs/overview.png)
![Investigate first](docs/investigate.png)
-->

## Methodology notes
- **No data leakage:** features use only information known at arrival (rolling department waits are shifted to exclude the current visit; no service or discharge times are used as inputs).
- **Delay targets are an assumption:** Critical 15, High 30, Medium 45, Low 60 minutes. Edit the `SLA` dictionary at the top of `src/train_queuesight.py`.
- **Data validation:** no nulls or duplicate encounters; `wait_minutes` verified against timestamps on every row; "Unknown" categories kept as their own value.
- **Capacity columns** in `departments.csv` are far larger than the daily visits in the sample, so they are not used as literal capacity limits.
- **Small-sample alerts** are shown with a confidence score and should be read as a watchlist, not a verdict.

## Limitations
- Data is simulated, so findings are not clinical truth and associations are not causal.
- The signal is mostly triage × department; richer real-world data (staffing, bed occupancy, queue length) would likely improve predictions.
- Delay targets are illustrative, not hospital standards.
- Weekly department alerts can occasionally be chance variation when many departments and weeks are scanned.

## Roadmap
- [ ] Demand forecasting by department, day and hour (with seasonal-naive baseline)
- [ ] SHAP-style per-alert explanations
- [ ] What-if capacity simulation (add staff or beds, see predicted wait change)
- [ ] Week-over-week alert history and trends
- [ ] Validation with real hospital data and clinical input

## Deploying the dashboard
1. Push this folder to a public GitHub repository (include the `out/` folder).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and choose **Create app**.
3. Select the repository, branch `main`, and main file path `app/app.py`.
4. Click **Deploy**.

## Tech stack
Python · pandas · NumPy · scikit-learn · SciPy · joblib · Streamlit · Plotly · Matplotlib




