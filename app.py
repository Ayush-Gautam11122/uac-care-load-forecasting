"""UAC Program – Care Load & Placement Demand Forecasting dashboard.
Run:  streamlit run app.py
Reads pre-computed artefacts from ./results (produced by src/forecast.py + src/metrics.py)."""
import json, pathlib
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go

R = pathlib.Path(__file__).parent / "results"
MODELS = ["Naive", "MovingAvg7", "ARIMA", "SARIMA", "ETS", "RandomForest", "GradientBoosting"]
COL = {"Naive": "#9e9e9e", "MovingAvg7": "#b59f3b", "ARIMA": "#2a9d8f", "SARIMA": "#1b6f66",
       "ETS": "#8e6bbf", "RandomForest": "#e07a1f", "GradientBoosting": "#c0392b"}
st.set_page_config(page_title="UAC Care-Load Forecasting", layout="wide", page_icon="📈")

# ---------------------------------------------------------------- data loading
@st.cache_data
def load():
    d = pd.read_csv(R / "data_clean.csv", index_col=0, parse_dates=True)
    out = {"d": d, "summary": json.load(open(R / "summary.json"))}
    for t in ("hhs_care", "discharged"):
        out[f"final_{t}"] = pd.read_csv(R / f"final_{t}.csv", parse_dates=["date"])
        out[f"iv_{t}"] = pd.read_csv(R / f"intervals_{t}.csv")
        for tag in ("main", "stress"):
            out[f"met_{t}_{tag}"] = pd.read_csv(R / f"metrics_{t}_{tag}.csv")
        bt = pd.read_csv(R / f"bt_{t}_main.csv", parse_dates=["origin", "target_date"])
        bt["e"] = bt.y_true - bt.y_pred
        out[f"bt_{t}"] = bt
    out["kpis"] = pd.read_csv(R / "kpis.csv")
    out["cov"] = pd.read_csv(R / "coverage95_hhs_care.csv", index_col=0).iloc[:, 0]
    return out

D = load(); d = D["d"]; best = D["summary"]["best"]
last_date = d.index[-1]; last_care = d.hhs_care.iloc[-1]

@st.cache_data
def resid_samples(target, model, h):
    bt = D[f"bt_{target}"]
    x = bt[(bt.model == model) & (bt.h == h) & (bt.observed == 1)].e.values
    return x

@st.cache_data
def cum_resid(model, H):
    """Error of the H-day cumulative discharge forecast, per forecast origin (backtest)."""
    bt = D["bt_discharged"]; x = bt[(bt.model == model) & (bt.h <= H)]
    g = x.groupby("origin").agg(n=("h", "size"), e=("e", "sum"))
    return g[g.n == H].e.values

def fc(target, model):
    f = D[f"final_{target}"]; return f[f.model == model].sort_values("h").set_index("h")

def band(target, model, level):
    iv = D[f"iv_{target}"]; iv = iv[iv.model == model].set_index("h")
    lo, hi = ("q025", "q975") if level == 95 else ("q10", "q90")
    p = fc(target, model).forecast
    return p + iv[lo], p + iv[hi]

# -------------------------------------------------------------------- sidebar
st.sidebar.title("Forecast controls")
H = st.sidebar.slider("Forecast horizon (days)", 1, 14, 14)
sel = st.sidebar.multiselect("Models to show", MODELS, default=[best["hhs_care"], "Naive"])
if not sel: sel = [best["hhs_care"]]
primary = st.sidebar.selectbox("Primary model (confidence band & risk)", sel, index=0)
level = st.sidebar.radio("Confidence interval", [80, 95], index=1, horizontal=True)
cap = st.sidebar.number_input("Capacity threshold (children in care)", 1000, 15000,
                              2600, step=50,
                              help="Used for the capacity-breach probability. Default 2,600 is illustrative (≈ top of the 14-day 95% band); the KPI backtest used 2,450. Replace with HHS's actual operating capacity.")
hist_days = st.sidebar.slider("History shown (days)", 30, 365, 120)
st.sidebar.caption(f"Data through **{last_date:%d %b %Y}** · last reported load **{last_care:,.0f}**")

st.title("Predictive forecasting of care load & placement demand")
st.caption("HHS Unaccompanied Children (UC) Program · daily reporting Jan 2023 – Dec 2025 · walk-forward validated models")

pf = fc("hhs_care", primary)
lo, hi = band("hhs_care", primary, level)
f_h = pf.forecast.loc[H]
res = resid_samples("hhs_care", primary, H)
p_breach = float((f_h + res >= cap).mean())
pd_f = fc("discharged", best["discharged"]); dis7 = pd_f.forecast.loc[:min(H, 14)].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Care load in {H} d ({primary})", f"{f_h:,.0f}", f"{f_h-last_care:+,.0f} vs today")
c2.metric(f"{level}% interval", f"{lo.loc[H]:,.0f} – {hi.loc[H]:,.0f}")
c3.metric(f"P(load ≥ {cap:,}) at day {H}", f"{p_breach:.0%}")
c4.metric(f"Expected discharges, next {H} d", f"{dis7:,.0f}", help=f"{best['discharged']} model")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Care-load forecast", "Discharge demand", "Model comparison", "Scenarios", "Data & method"])

# ------------------------------------------------------------------- tab 1
with tab1:
    h0 = d.hhs_care.iloc[-hist_days:]
    fig = go.Figure()
    fig.add_scatter(x=h0.index, y=h0, name="Actual", line=dict(color="#1f4e79", width=2))
    dates = pf.index.map(lambda h: last_date + pd.Timedelta(days=int(h)))
    m = pf.index <= H
    fig.add_scatter(x=list(dates[m]) + list(dates[m])[::-1], y=list(hi[m]) + list(lo[m])[::-1], fill="toself",
                    fillcolor="rgba(224,122,31,0.18)", line=dict(width=0), name=f"{level}% interval ({primary})", hoverinfo="skip")
    for mname in sel:
        p = fc("hhs_care", mname).forecast
        fig.add_scatter(x=[last_date] + list(dates[m]), y=[last_care] + list(p[m]), name=mname,
                        line=dict(color=COL[mname], dash="solid" if mname == primary else "dot", width=3 if mname == primary else 2))
    fig.add_hline(y=cap, line_dash="dash", line_color="#c0392b", annotation_text="capacity threshold")
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Children in HHS care", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    # breach probability profile
    prof = [{"Day": h, "Date": (last_date + pd.Timedelta(days=h)).date(), "Forecast": pf.forecast.loc[h],
             "Lower": band("hhs_care", primary, level)[0].loc[h], "Upper": band("hhs_care", primary, level)[1].loc[h],
             "P(breach)": float((pf.forecast.loc[h] + resid_samples("hhs_care", primary, h) >= cap).mean())} for h in range(1, H + 1)]
    prof = pd.DataFrame(prof)
    cA, cB = st.columns([3, 2])
    cA.dataframe(prof.style.format({"Forecast": "{:,.0f}", "Lower": "{:,.0f}", "Upper": "{:,.0f}", "P(breach)": "{:.0%}"}), hide_index=True, width="stretch")
    pk = prof.loc[prof["P(breach)"].idxmax()]
    status = "🔴 Elevated" if pk["P(breach)"] >= .5 else "🟠 Watch" if pk["P(breach)"] >= .2 else "🟢 Normal"
    cB.markdown(f"### Early-warning status: {status}\nPeak breach probability **{pk['P(breach)']:.0%}** on day {int(pk.Day)} ({pk.Date}). "
                "Bands are empirical walk-forward error quantiles, so they reflect how wrong this model actually was at each lead time in 2025.")

# ------------------------------------------------------------------- tab 2
with tab2:
    dm = st.selectbox("Discharge model", MODELS, index=MODELS.index(best["discharged"]))
    pdv = fc("discharged", dm).forecast.loc[:H]; iv = D["iv_discharged"]; iv = iv[iv.model == dm].set_index("h")
    lo_d, hi_d = ((pdv + iv.q025.loc[:H]), (pdv + iv.q975.loc[:H])) if level == 95 else ((pdv + iv.q10.loc[:H]), (pdv + iv.q90.loc[:H]))
    dts = [last_date + pd.Timedelta(days=int(h)) for h in pdv.index]
    hd = d.discharged.iloc[-hist_days:]
    fig = go.Figure()
    fig.add_scatter(x=hd.index, y=hd, name="Actual (daily)", line=dict(color="#9bb7d4", width=1))
    fig.add_scatter(x=hd.index, y=hd.rolling(7).mean(), name="Actual 7-day avg", line=dict(color="#1f4e79", width=2))
    fig.add_scatter(x=dts + dts[::-1], y=list(hi_d) + list(lo_d)[::-1], fill="toself", fillcolor="rgba(192,57,43,0.15)", line=dict(width=0), name=f"{level}% interval", hoverinfo="skip")
    fig.add_scatter(x=dts, y=pdv, name=f"Forecast ({dm})", line=dict(color="#c0392b", width=3))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Discharges per day", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    Hs = [x for x in (7, 14) if x <= H] or [H]
    cols = st.columns(len(Hs) + 1)
    for c, hh in zip(cols, Hs):
        tot = fc("discharged", dm).forecast.loc[:hh].sum(); ce = cum_resid(dm, hh)
        qlo, qhi = np.quantile(ce, [.025, .975] if level == 95 else [.10, .90])
        c.metric(f"Cumulative discharges, next {hh} d", f"{tot:,.0f}", f"{level}%: {tot+qlo:,.0f} – {tot+qhi:,.0f}", delta_color="off")
    tr = d.transferred.iloc[-14:].mean(); tot = fc("discharged", dm).forecast.loc[:H].sum()
    cols[-1].metric(f"Placement balance, next {H} d", f"{tr*H - tot:+,.0f}", help=f"Expected transfers into HHS (recent 14-d mean {tr:.1f}/day × {H}) minus forecast discharges. Positive = intake outpacing placements.")
    st.caption("Cumulative intervals use the backtest distribution of summed errors, so day-to-day error correlation is respected. Placement balance is a flow signal only: in this dataset daily transfers − discharges explains little of the change in stock (r≈0.2–0.35), because other exits/entries are not reported.")

# ------------------------------------------------------------------- tab 3
with tab3:
    c1, c2 = st.columns([1, 1])
    tgt = c1.radio("Target", ["hhs_care", "discharged"], format_func=lambda x: {"hhs_care": "Children in HHS care", "discharged": "Daily discharges"}[x], horizontal=True)
    win = c2.radio("Evaluation window", ["main", "stress"], format_func=lambda x: {"main": "Main test: Apr–Dec 2025 (stable regime)", "stress": "Stress test: 15 Dec 2024–Mar 2025 (rapid decline)"}[x], horizontal=True)
    met = D[f"met_{tgt}_{win}"]; metric = st.selectbox("Metric", ["MAE", "RMSE", "MAPE"])
    fig = go.Figure()
    for mname in MODELS:
        x = met[met.model == mname]; fig.add_scatter(x=x.h, y=x[metric], name=mname, line=dict(color=COL[mname], width=3 if mname in sel else 1.5), opacity=1 if mname in sel else .55)
    fig.update_layout(height=380, xaxis_title="Forecast horizon (days)", yaxis_title=metric + (" (%)" if metric == "MAPE" else ""), margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    tab = met[met.h.isin([1, 3, 7, 14])].pivot(index="model", columns="h", values=metric).reindex(MODELS)
    tab.columns = [f"h={c}" for c in tab.columns]; tab["Mean h1–14"] = met.groupby("model")[metric].mean().reindex(MODELS)
    st.dataframe(tab.style.format("{:,.1f}").highlight_min(axis=0, color="#cfe8cf"), width="stretch")
    st.subheader("Early-warning KPIs for care load (7-day horizon)")
    k = D["kpis"].set_index("model").reindex(MODELS)
    k = k.assign(coverage95=D["cov"].reindex(MODELS))
    k.columns = ["Forecast accuracy % (100−MAPE)", "MAE", "Stability index", "Brier (breach)", "Skill vs naive: main", "Skill vs naive: stress",
                 "Surge lead time (d)", "Breaches detected", "Breach events", "Precision", "Recall", "95% PI coverage (OOS)"]
    st.dataframe(k.style.format({c: "{:.2f}" for c in k.columns}).format({"MAE": "{:.1f}", "Forecast accuracy % (100−MAPE)": "{:.1f}", "Breaches detected": "{:.0f}", "Breach events": "{:.0f}"}), width="stretch")
    st.caption("Stability index = 1 − CV of monthly MAE (higher = steadier). Skill = 1 − MAE/MAE(naive); negative means worse than persistence. Lead time = largest forecast lead (≤14 d) at which the model already predicted a threshold crossing, averaged over crossings.")

# ------------------------------------------------------------------- tab 4
with tab4:
    st.markdown("**What-if analysis.** Adjust inflow and discharge pace relative to the baseline forecast and compare resulting care-load paths. "
                "Changes in net flow are added cumulatively to the baseline forecast.")
    s1, s2, s3 = st.columns(3)
    t_up = s1.slider("Transfers into HHS vs recent average", -50, 300, 100, step=10, format="%d%%", help="100% = double the recent 14-day average")
    d_dn = s2.slider("Discharge pace vs forecast", -60, 60, -25, step=5, format="%d%%")
    pt = s3.slider("Pass-through to stock", 0.0, 1.0, 0.8, 0.1, help="Share of a net-flow change that shows up in the HHS census. Regressing 7-day census change on the prior week's net flow (2025 regime) gives ≈0.8 (overlapping windows, so indicative); daily coupling is much weaker.")
    base = fc("hhs_care", primary).forecast.loc[:H]; dis = fc("discharged", best["discharged"]).forecast.loc[:H]
    tr = d.transferred.iloc[-14:].mean()
    scen = {"Baseline": (0, 0), "Intake surge": (t_up / 100, 0), "Slower discharges": (0, d_dn / 100), "Surge + slower discharges": (t_up / 100, d_dn / 100)}
    fig = go.Figure(); rows = []
    for nm, (ti, di) in scen.items():
        dn = ti * tr - di * dis.values            # extra net inflow per day
        path = base.values + pt * np.cumsum(dn)
        fig.add_scatter(x=[last_date + pd.Timedelta(days=int(h)) for h in base.index], y=path, name=nm, line=dict(width=3 if nm == "Baseline" else 2))
        rows.append({"Scenario": nm, f"Load day {H}": path[-1], "vs baseline": path[-1] - base.values[-1], "Peak": path.max(), "Days ≥ capacity": int((path >= cap).sum())})
    fig.add_hline(y=cap, line_dash="dash", line_color="#c0392b", annotation_text="capacity threshold")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Children in HHS care", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows).style.format({f"Load day {H}": "{:,.0f}", "vs baseline": "{:+,.0f}", "Peak": "{:,.0f}"}), hide_index=True, width="stretch")
    # staffing lens
    ratio = st.number_input("Children per caseworker (planning ratio)", 1, 50, 12)
    peak = max(r["Peak"] for r in rows)
    st.info(f"At {ratio}:1, the baseline requires **{np.ceil(rows[0]['Peak']/ratio):,.0f}** caseworkers vs **{np.ceil(peak/ratio):,.0f}** in the most severe scenario "
            f"(Δ {np.ceil(peak/ratio)-np.ceil(rows[0]['Peak']/ratio):,.0f}). Illustrative ratio – set to actual policy.")

# ------------------------------------------------------------------- tab 5
with tab5:
    st.markdown(f"""
**Data.** {int(d.observed.sum())} reported days, Jan 12 2023 – Dec 21 2025, reindexed to {len(d):,} calendar days. **{(1-d.observed.mean()):.0%}** of days were not reported
(reports cover Sun–Thu only; Fri/Sat are never present) and were linearly interpolated. Models are scored **only on days with a real report**.

**Quality flags.** 374 rows carry flow values (intake / CBP custody / transfers) that exactly repeat other rows, concentrated in 2023–Jan 2024, which looks like copy-paste in the source file. HHS care and discharges differ across those rows and were kept; flow-based features from those periods deserve caution.

**Regime change.** Load fell from ~11,500 (Dec 2023) to ~2,000–2,500 (since Mar 2025). Forecast errors are therefore reported on the **post-Apr 2025 regime** (main) and, separately, on the **Dec 2024–Mar 2025 decline** (stress).

**Models.** Naive persistence · 7-day moving average · ARIMA(1,1,1) · SARIMA(1,1,1)(1,0,1)₇ · damped-trend exponential smoothing · Random Forest · Gradient Boosting (direct, one model per horizon, predicting the change from today's level; features: lags 1/7/14, 7/14-day rolling means & variances, net pressure, intake, CBP load, weekday, month, US-holiday flag).
Expanding-window walk-forward; models refit every 21 days; training pairs never include targets after the forecast origin.

**Intervals.** Empirical quantiles of walk-forward residuals per model and horizon (not Gaussian assumptions). Breach probability = share of those residuals that would push the forecast across the capacity threshold.
""")
    with st.expander("Cleaned daily data"): st.dataframe(d.tail(120))
