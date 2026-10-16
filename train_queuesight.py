"""QueueSight - end-to-end pipeline: load -> join -> features -> train (from scratch) -> evaluate (time-aware) -> alerts.
Usage (run from the project root folder): python src/train_queuesight.py data
Outputs (./out): metrics.json, priority_queue.csv, weekly_watchlist.csv, wait_model.joblib, breach_model.joblib
"""
import sys, json, pathlib, numpy as np, pandas as pd, joblib
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data"); OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
SLA = {"Critical": 15, "High": 30, "Medium": 45, "Low": 60}   # ASSUMPTION: configurable delay targets (min)

# ---------- 1. LOAD + JOIN + CLEAN ----------
e = pd.read_csv(SRC/"encounters.csv", parse_dates=["arrival_time","service_start_time","service_end_time","discharge_time"])
d = pd.read_csv(SRC/"departments.csv"); p = pd.read_csv(SRC/"patients.csv", keep_default_na=False)
t = pd.read_csv(SRC/"diagnostic_tests.csv", parse_dates=["requested_time","completion_time"])
assert not e.encounter_id.duplicated().any()
assert (abs((e.service_start_time-e.arrival_time).dt.total_seconds()/60 - e.wait_minutes) <= 1).all()
e = e.merge(d, on="department_id").merge(p, on="patient_id", how="left").sort_values("arrival_time").reset_index(drop=True)

# ---------- 2. LEAK-FREE FEATURES (only information known at arrival) ----------
e["hour"] = e.arrival_time.dt.hour; e["dow"] = e.arrival_time.dt.dayofweek
e["dept_prev_wait20"] = e.groupby("department_id").wait_minutes.transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
arr = np.zeros(len(e))
for _, idx in e.groupby("department_id").groups.items():            # arrivals in the previous 24h, same department
    ts = e.loc[idx, "arrival_time"].values.astype("datetime64[m]").astype(np.int64)
    arr[idx] = np.searchsorted(ts, ts, "left") - np.searchsorted(ts, ts-1440, "left")
e["dept_arrivals_prev24h"] = arr
e["breach"] = (e.wait_minutes > e.triage_level.map(SLA)).astype(int)
for c in ["department_id","visit_type","triage_level"]: e[c] = e[c].astype("category")
F = ["triage_level","department_id","visit_type","hour","dept_prev_wait20","dept_arrivals_prev24h"]

# ---------- 3. TIME-AWARE SPLIT ----------
tr = e[e.arrival_time < "2025-09-01"]; va = e[(e.arrival_time >= "2025-09-01") & (e.arrival_time < "2025-11-01")]; te = e[e.arrival_time >= "2025-11-01"]
trva = pd.concat([tr, va]); M = {}
mk = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_depth=3, categorical_features="from_dtype", random_state=42)

# baselines
M["baseline_global_MAE"] = mean_absolute_error(te.wait_minutes, [tr.wait_minutes.mean()]*len(te))
tm = tr.groupby("triage_level", observed=True).wait_minutes.mean(); M["baseline_triage_MAE"] = mean_absolute_error(te.wait_minutes, te.triage_level.map(tm).astype(float))
dt = tr.groupby(["department_id","triage_level"], observed=True).wait_minutes.mean()
M["baseline_dept_x_triage_MAE"] = mean_absolute_error(te.wait_minutes, te.set_index(["department_id","triage_level"]).index.map(dt).values)

# ---------- 4. MODELS ----------
m = mk().fit(trva[F], trva.wait_minutes); pr = m.predict(te[F])
M.update(gbm_MAE=mean_absolute_error(te.wait_minutes, pr), gbm_R2=r2_score(te.wait_minutes, pr))
M["MAE_reduction_vs_global_pct"] = 100*(1-M["gbm_MAE"]/M["baseline_global_MAE"])
res = (trva.wait_minutes - m.predict(trva[F])).values; lo, hi = np.quantile(res, [.05, .95])        # 90% prediction interval
M.update(interval_low=lo, interval_high=hi, interval_coverage_test=float(((te.wait_minutes >= pr+lo) & (te.wait_minutes <= pr+hi)).mean()))
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_depth=3, categorical_features="from_dtype", random_state=42).fit(trva[F], trva.breach)
pp = clf.predict_proba(te[F])[:, 1]; M.update(breach_rate_test=te.breach.mean(), breach_AUC=roc_auc_score(te.breach, pp), breach_AP=average_precision_score(te.breach, pp))
pi = permutation_importance(m, te[F], te.wait_minutes, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
M["permutation_importance_MAE_min"] = dict(zip(F, pi.importances_mean.round(3)))

# ---------- 5. DIAGNOSTIC TEST TURNAROUND MONITOR ----------
t["tat"] = (t.completion_time-t.requested_time).dt.total_seconds()/60
M["tests"] = dict(median_tat=t.tat.median(), pct_over_120=float((t.tat > 120).mean()),
                  urgent_mean=t[t.priority=="Urgent"].tat.mean(), routine_mean=t[t.priority=="Routine"].tat.mean(),
                  urgent_pct_over_60=float((t[t.priority=="Urgent"].tat > 60).mean()))

# ---------- 6. ALERTS: "what should we investigate first?" = impact x confidence ----------
cut = e.arrival_time.max() - pd.Timedelta(days=28); hist = e[e.arrival_time < cut]; recent = e[e.arrival_time >= cut].copy()
mh = mk().fit(hist[F], hist.wait_minutes); recent["resid"] = recent.wait_minutes - mh.predict(recent[F])   # worse than expected for this dept x triage
sigma = (hist.wait_minutes - mh.predict(hist[F])).std()
q = recent.groupby("department_name").agg(patients=("resid","size"), avg_excess_min=("resid","mean"), actual_wait=("wait_minutes","mean")).reset_index()
q["z"] = q.avg_excess_min*np.sqrt(q.patients)/sigma; q["confidence"] = norm.cdf(q.z); q["impact_min"] = q.avg_excess_min*q.patients
q["priority"] = (q.impact_min.clip(lower=0)*q.confidence).round(1)
top_driver = recent.groupby(["department_name","triage_level"], observed=True).resid.mean().reset_index().sort_values("resid", ascending=False).drop_duplicates("department_name")
q = q.merge(top_driver.rename(columns={"triage_level":"driver_triage","resid":"driver_excess"}), on="department_name").sort_values("priority", ascending=False)
q.round(2).to_csv(OUT/"priority_queue.csv", index=False)
recent[["encounter_id","department_name","triage_level","wait_minutes","resid"]].round(2).to_csv(OUT/"recent_residuals.csv", index=False)   # used by the app's explanation panel
# weekly department watchlist (rolling z vs expectation, whole year, for the trend view)
e["resid_all"] = e.wait_minutes - mk().fit(trva[F], trva.wait_minutes).predict(e[F]); e["week"] = e.arrival_time.dt.to_period("W").astype(str)
w = e.groupby(["department_name","week"]).resid_all.agg(["mean","size"]).reset_index(); w = w[w["size"] >= 8]; w["z"] = w["mean"]*np.sqrt(w["size"])/sigma
w.sort_values("z", ascending=False).head(25).round(2).to_csv(OUT/"weekly_watchlist.csv", index=False)

joblib.dump(m, OUT/"wait_model.joblib"); joblib.dump(clf, OUT/"breach_model.joblib")
json.dump(M, open(OUT/"metrics.json","w"), indent=2, default=float); print(json.dumps(M, indent=1, default=float)); print(q.round(1).head(6).to_string())
