"""STEP 1 - Look at the data before modelling. Run from the project root:  python src/01_explore_data.py
Prints data-quality checks and saves 3 charts into out/figures/. Nothing here changes the data."""
import pandas as pd, pathlib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = pathlib.Path("data"); FIG = pathlib.Path("out/figures"); FIG.mkdir(parents=True, exist_ok=True)
enc = pd.read_csv(D/"encounters.csv", parse_dates=["arrival_time","service_start_time"])
dep = pd.read_csv(D/"departments.csv"); tests = pd.read_csv(D/"diagnostic_tests.csv", parse_dates=["requested_time","completion_time"])
df = enc.merge(dep, on="department_id")            # JOIN: attach department name to every visit

print("Rows:", len(enc), "| missing values:", int(enc.isna().sum().sum()), "| duplicate visit IDs:", int(enc.encounter_id.duplicated().sum()))
calc = (enc.service_start_time - enc.arrival_time).dt.total_seconds()/60
print("Rows where wait_minutes disagrees with timestamps:", int((abs(calc-enc.wait_minutes) > 1).sum()))
print("\nAverage wait by triage level (minutes):\n", df.groupby("triage_level").wait_minutes.mean().round(1).sort_values())
print("\nAverage wait by department (minutes):\n", df.groupby("department_name").wait_minutes.mean().round(1).sort_values())
tests["tat"] = (tests.completion_time - tests.requested_time).dt.total_seconds()/60
print("\nTest turnaround by priority (minutes):\n", tests.groupby("priority").tat.mean().round(1))

df.groupby("triage_level").wait_minutes.mean().reindex(["Critical","High","Medium","Low"]).plot.bar(color="#19E3F5", edgecolor="black", title="Average wait by triage level (min)")
plt.tight_layout(); plt.savefig(FIG/"wait_by_triage.png", dpi=150); plt.clf()
df[df.triage_level=="High"].groupby("department_name").wait_minutes.mean().sort_values().plot.barh(color="#19E3F5", edgecolor="black", title="Same acuity, different wait: HIGH-triage patients (min)")
plt.tight_layout(); plt.savefig(FIG/"high_triage_by_department.png", dpi=150); plt.clf()
df.groupby(df.arrival_time.dt.hour).wait_minutes.mean().plot(color="black", title="Average wait by hour of arrival (flat = not a rush-hour problem)", ylim=(0,70))
plt.tight_layout(); plt.savefig(FIG/"wait_by_hour.png", dpi=150); print("\nCharts saved in out/figures/")
