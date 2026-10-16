"""QueueSight dashboard.  Run from the project root:  streamlit run app/app.py"""
import json, pathlib, joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
ROOT = pathlib.Path(__file__).resolve().parent.parent; D = ROOT/"data"; OUT = ROOT/"out"
SLA = {"Critical": 15, "High": 30, "Medium": 45, "Low": 60}; CYAN = "#19E3F5"
st.set_page_config(page_title="QueueSight", page_icon="🏥", layout="wide")

if not (OUT/"metrics.json").exists():
    st.error("Model files not found. In your terminal run:  python src/train_queuesight.py data   then refresh this page."); st.stop()

@st.cache_data
def load():
    e = pd.read_csv(D/"encounters.csv", parse_dates=["arrival_time"]).merge(pd.read_csv(D/"departments.csv"), on="department_id")
    t = pd.read_csv(D/"diagnostic_tests.csv", parse_dates=["requested_time","completion_time"])
    t["tat"] = (t.completion_time - t.requested_time).dt.total_seconds()/60
    return e.sort_values("arrival_time").reset_index(drop=True), t
@st.cache_resource
def models(): return joblib.load(OUT/"wait_model.joblib"), joblib.load(OUT/"breach_model.joblib")
e, t = load(); wait_model, breach_model = models(); M = json.load(open(OUT/"metrics.json"))
queue = pd.read_csv(OUT/"priority_queue.csv"); resid = pd.read_csv(OUT/"recent_residuals.csv")

st.title("🏥 QueueSight")
st.caption("What is likely to get delayed next, why, and what should operations examine?  ·  Synthetic data – not for clinical use.")
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predict a wait", "Investigate first", "Test monitor"])

with tab1:
    tri = st.multiselect("Filter by triage level", ["Critical","High","Medium","Low"], default=["Critical","High","Medium","Low"])
    f = e[e.triage_level.isin(tri)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Visits", f"{len(f):,}"); c2.metric("Average wait", f"{f.wait_minutes.mean():.0f} min")
    hi = e[e.triage_level.isin(["Critical","High"])]; c3.metric("Critical/High waiting > 30 min", f"{(hi.wait_minutes>30).mean()*100:.0f}%")
    c4.metric("Tests taking > 2 h", f"{M['tests']['pct_over_120']*100:.0f}%")
    a, b = st.columns(2)
    d = f.groupby("department_name").wait_minutes.mean().sort_values().reset_index()
    a.plotly_chart(px.bar(d, x="wait_minutes", y="department_name", orientation="h", title="Average wait by department (min)", color_discrete_sequence=[CYAN]))
    g = e.groupby("triage_level").wait_minutes.mean().reindex(["Critical","High","Medium","Low"]).reset_index()
    b.plotly_chart(px.bar(g, x="triage_level", y="wait_minutes", title="Average wait by triage level (min)", color_discrete_sequence=[CYAN]))
    st.info("Waits rise with triage level, which is expected. QueueSight therefore compares each service with what is normal for its acuity, not with raw averages.")

with tab2:
    st.subheader("Expected wait for a new arrival")
    names = e.drop_duplicates("department_id").set_index("department_name").department_id.to_dict()
    c1, c2, c3, c4 = st.columns(4)
    dn = c1.selectbox("Department", sorted(names)); tr = c2.selectbox("Triage level", ["Critical","High","Medium","Low"], index=1)
    vt = c3.selectbox("Visit type", sorted(e.visit_type.unique())); hr = c4.slider("Hour of arrival", 0, 23, 10)
    did = names[dn]; recent20 = e[e.department_id == did].wait_minutes.tail(20).mean()
    row = pd.DataFrame({"triage_level": pd.Categorical([tr], categories=sorted(e.triage_level.unique())),
                        "department_id": pd.Categorical([did], categories=sorted(e.department_id.unique())),
                        "visit_type": pd.Categorical([vt], categories=sorted(e.visit_type.unique())),
                        "hour": [hr], "dept_prev_wait20": [recent20], "dept_arrivals_prev24h": [float(e.groupby(["department_id", e.arrival_time.dt.date]).size().mean())]})
    pred = float(wait_model.predict(row)[0]); lo, hi_ = pred + M["interval_low"], pred + M["interval_high"]
    p_breach = float(breach_model.predict_proba(row)[0, 1])
    k1, k2, k3 = st.columns(3); k1.metric("Expected wait", f"{pred:.0f} min"); k2.metric("90% range", f"{max(lo,0):.0f} – {hi_:.0f} min")
    k3.metric(f"Risk of exceeding {SLA[tr]}-min target", f"{p_breach*100:.0f}%")
    st.caption(f"Model error on unseen Nov–Dec data: about ±{M['gbm_MAE']:.0f} min on average. Delay targets are an assumption (Critical 15 · High 30 · Medium 45 · Low 60).")

with tab3:
    st.subheader("What should we investigate first?  (last 28 days)")
    st.caption("Priority = extra waiting minutes beyond what is expected for that department's acuity mix × confidence that it is real.")
    show = queue.rename(columns={"department_name":"Department","patients":"Patients","avg_excess_min":"Avg excess (min)","confidence":"Confidence","impact_min":"Impact (min)","priority":"Priority","driver_triage":"Main driver"})
    st.dataframe(show[["Department","Patients","Avg excess (min)","Confidence","Impact (min)","Priority","Main driver"]], hide_index=True)
    pick = st.selectbox("Why was this flagged?", queue.department_name)
    r = queue[queue.department_name == pick].iloc[0]
    a, b = st.columns([1, 1.4])
    a.markdown(f"**{pick}**\n\n- {int(r.patients)} patients in the window\n- Waited **{r.avg_excess_min:+.1f} min** vs expected on average\n- Signal strength z = **{r.z:.1f}** → confidence **{r.confidence*100:.0f}%**\n- Biggest contributor: **{r.driver_triage}**-triage patients ({r.driver_excess:+.1f} min)\n- Total impact ≈ **{r.impact_min:.0f}** patient-minutes")
    rr = resid[resid.department_name == pick].groupby("triage_level").resid.mean().reindex(["Critical","High","Medium","Low"]).reset_index()
    b.plotly_chart(px.bar(rr, x="triage_level", y="resid", title="Excess wait by triage level (min vs expected)", color_discrete_sequence=[CYAN]))
    if r.confidence < 0.9: st.warning("Low-to-moderate confidence: treat as a watchlist item, not a verdict.")

with tab4:
    st.subheader("Diagnostic-test turnaround")
    k1, k2, k3 = st.columns(3); k1.metric("Median turnaround", f"{t.tat.median():.0f} min"); k2.metric("Urgent tests over 60 min", f"{M['tests']['urgent_pct_over_60']*100:.0f}%")
    k3.metric("Urgent vs Routine (avg)", f"{M['tests']['urgent_mean']:.0f} vs {M['tests']['routine_mean']:.0f} min")
    s = t.groupby("test_type").tat.agg(avg="mean", p90=lambda x: x.quantile(.9)).round(0).reset_index().melt("test_type", var_name="measure", value_name="minutes")
    st.plotly_chart(px.bar(s, x="test_type", y="minutes", color="measure", barmode="group", title="Turnaround by test type (min)", color_discrete_sequence=[CYAN, "#111111"]))
    st.warning("Priority inversion: Urgent tests take about as long as Routine ones, so the Urgent label is not speeding anything up.")
