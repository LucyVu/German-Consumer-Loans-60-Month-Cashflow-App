"""
Created on Wed Aug 27 23:52:46 2025

@author: LucyVu
"""
import pandas as pd
import numpy as np
import os
import io
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Portfolio Cash Flow Projection", layout="wide")
st.title("Germany Loan Portfolio Cash Flow Projection")

# ---------- Helpers ----------
def annuity_payment(balance: float, r_m: float, n: int) -> float:
    """Compute annuity payment when monthly rate or term might be edge-case."""
    if n <= 0:
        return 0.0
    if abs(r_m) < 1e-12:
        return balance / n
    return balance * (r_m / (1 - (1 + r_m) ** (-n)))

def cpr_to_smm(cpr: float) -> float:
    """Convert CPR (annual) to SMM (monthly)."""
    return 1 - (1 - cpr) ** (1/12)

def annual_pd_to_monthly(pd_annual: float) -> float:
    """Convert annual PD (0..1) to a monthly default probability."""
    pd_annual = max(0.0, min(pd_annual, 1.0))
    return 1 - (1 - pd_annual) ** (1/12)

def read_loan_tape(path_or_buffer, sheet_name: str = "loan_tape") -> pd.DataFrame:
    """
    Read a loan tape from CSV or Excel. Uses openpyxl for .xlsx and shows
    clear errors if dependency/sheet is missing.
    """
    name = getattr(path_or_buffer, "name", None)
    path_str = str(path_or_buffer) if isinstance(path_or_buffer, str) else (name or "")

    # CSV path or uploaded CSV buffer
    if path_str.lower().endswith(".csv"):
        return pd.read_csv(path_or_buffer)

    # Excel path or uploaded Excel buffer
    try:
        return pd.read_excel(path_or_buffer, sheet_name=sheet_name, engine="openpyxl")
    except ImportError:
        st.error(
            "Excel support requires the 'openpyxl' package.\n\n"
            "Fix: add `openpyxl` to requirements.txt and redeploy, "
            "or upload a CSV instead."
        )
        st.stop()
    except FileNotFoundError:
        st.error(f"File not found: {path_or_buffer}")
        st.stop()
    except ValueError as e:
        st.error(f"{e}\nMake sure your Excel has a sheet named '{sheet_name}'.")
        st.stop()

# ---------- Sidebar: data & assumptions ----------
st.sidebar.header("Data source")
file = st.sidebar.file_uploader("Upload data file", type=["xlsx", "csv"])
use_sample = st.sidebar.checkbox("Use bundled sample file: 'loanslite for app.xlsx'", value=True)
SAMPLE_PATH = "loanslite for app.xlsx"
sheet_name = st.sidebar.text_input("Excel sheet name", value="loan_tape")

st.sidebar.header("PD unit from tape")
pd_unit = st.sidebar.radio(
    "How is PD provided in the tape?",
    ["Annual (convert to monthly)", "Monthly already"],
    index=0
)

st.sidebar.header("Assumptions")
CPR_annual = st.sidebar.number_input("CPR (annual)", value=0.08, min_value=0.0, step=0.01, format="%.4f")
SMM_base = cpr_to_smm(CPR_annual)
servicing_bps = st.sidebar.number_input("Servicing fee (bps/year)", value=50, min_value=0, step=5)
disc_rate = st.sidebar.number_input("Discount rate (annual)", value=0.07, min_value=0.0, step=0.01, format="%.4f")
recovery_lag_m = st.sidebar.number_input("Recovery lag (months)", value=12, min_value=0, step=1)
months = st.sidebar.slider("Projection horizon (months)", 1, 60, 60)

st.sidebar.header("Scenarios & Overrides")
scenario = st.sidebar.selectbox("Scenario", ["Base", "Downside", "Upside"])
PD_multiplier = st.sidebar.number_input("PD multiplier (k)", value=1.0, step=0.1, format="%.2f")
LGD_shift = st.sidebar.number_input("LGD shift (±Δ)", value=0.0, step=0.05, format="%.2f")
CPR_multiplier = st.sidebar.number_input("CPR multiplier (m)", value=1.0, step=0.1, format="%.2f")
include_recoveries_in_wal = st.sidebar.checkbox("Include Recoveries in WAL (optional)", value=False)

# Scenario presets
if scenario == "Downside":
    PD_multiplier = max(PD_multiplier, 1.5)
    LGD_shift = max(LGD_shift, 0.10)
    CPR_multiplier = min(CPR_multiplier, 0.7)
    disc_rate = disc_rate + 0.02
elif scenario == "Upside":
    PD_multiplier = min(PD_multiplier, 0.7)
    LGD_shift = min(LGD_shift, -0.10)
    CPR_multiplier = max(CPR_multiplier, 1.3)

SMM_eff = SMM_base * CPR_multiplier
fee_m = (servicing_bps / 10000.0) / 12.0
d_m = disc_rate / 12.0

# ---------- Load data ----------
loans = None

if file is not None:
    # Uploaded Excel/CSV
    loans = read_loan_tape(file, sheet_name=sheet_name)

elif use_sample:
    # Use bundled sample next to app.py
    if os.path.exists(SAMPLE_PATH):
        loans = read_loan_tape(SAMPLE_PATH, sheet_name=sheet_name)
    else:
        st.error(
            f"Sample file '{SAMPLE_PATH}' not found in the app folder. "
            "Either upload a file via the sidebar or commit the sample "
            "Excel/CSV to the repo root with exactly this name."
        )
        st.stop()

if loans is None:
    st.info("Upload a loan tape or untick 'Use bundled sample file' to run the model.")
    st.stop()

# Required schema
required_cols = [
    "loan_id",
    "opening_balance",
    "interest_rate (annual)",
    "remaining_term",
    "monthly_payment",
    "pd",
    "lgd",
]
missing = [c for c in required_cols if c not in loans.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---------- Core engine ----------
def project_cashflows(loans_df: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    rows = []
    for _, r in loans_df.iterrows():
        loan_id = r["loan_id"]
        bal = float(r["opening_balance"])
        apr = float(r["interest_rate (annual)"])
        r_m = apr / 12.0
        rem_term = int(r.get("remaining_term", horizon_months))
        pmt = float(r.get("monthly_payment", 0.0))
        if pmt <= 0.0:
            pmt = annuity_payment(bal, r_m, rem_term)

        # ---- PD monthly (correct handling) ----
        raw_pd = float(r["pd"])
        if pd_unit.startswith("Annual"):
            pd_m = annual_pd_to_monthly(raw_pd) * PD_multiplier
        else:
            pd_m = raw_pd * PD_multiplier
        pd_m = min(max(pd_m, 0.0), 1.0)

        lgd = min(max(float(r["lgd"]) + LGD_shift, 0.0), 1.0)

        # Delayed recoveries bucket
        recs = {}
        for t in range(1, horizon_months + 1):
            if bal < 1e-9:
                cash = recs.get(t, 0.0)
                pv = cash / ((1 + d_m) ** t)
                rows.append([t, loan_id, 0.0, 0.0, 0.0, 0.0, 0.0,
                             recs.get(t, 0.0), 0.0, 0.0, cash, pv, 0.0])
                continue

            # 1) Interest
            interest = bal * r_m
            # 2) Scheduled principal
            sched_prin = max(0.0, min(bal, pmt - interest))
            # 3) Prepayment
            prepay_base = max(0.0, bal - sched_prin)
            prepay = prepay_base * SMM_eff
            # 4) Default (expected) on survivors after prepay
            default_base = max(0.0, prepay_base - prepay)
            def_prin = default_base * pd_m
            # 5) Loss & Recovery (recovery comes later)
            loss = def_prin * lgd
            rec_amount = def_prin * (1 - lgd)
            lag_month = int(recovery_lag_m)
            recs[t + lag_month] = recs.get(t + lag_month, 0.0) + rec_amount
            # 6) Fees on beginning balance
            fees = bal * fee_m
            # 7) End balance
            end_bal = max(0.0, bal - sched_prin - prepay - def_prin)
            # 8) Cashflow (this month) & PV
            cash = interest + sched_prin + prepay + recs.get(t, 0.0) - fees
            pv = cash / ((1 + d_m) ** t)

            rows.append([t, loan_id, bal, interest, sched_prin, prepay, def_prin,
                         recs.get(t, 0.0), fees, end_bal, cash, pv, loss])

            bal = end_bal

    return pd.DataFrame(rows, columns=[
        "t", "loan_id", "Beg_Bal", "Interest", "SchedPrin", "Prepay", "DefaultPrin",
        "Recoveries", "Fees", "End_Bal", "Cashflow", "PV", "Loss"
    ])

# Run engine
cf = project_cashflows(loans, months)

# ---------- Aggregation: portfolio ----------
port = cf.groupby("t", as_index=False).agg({
    "Beg_Bal": "sum",
    "Interest": "sum",
    "SchedPrin": "sum",
    "Prepay": "sum",
    "DefaultPrin": "sum",
    "Recoveries": "sum",
    "Fees": "sum",
    "End_Bal": "sum",
    "Cashflow": "sum",
    "PV": "sum",
    "Loss": "sum",
})
port["CumLoss"] = port["Loss"].cumsum()

# ---------- KPIs ----------
# WAL (cash) — exclude defaults; optionally include recoveries
principal_cash = port["SchedPrin"] + port["Prepay"]
if include_recoveries_in_wal:
    principal_cash = principal_cash + port["Recoveries"]

total_principal_cash = principal_cash.sum()
WAL_years = 0.0 if total_principal_cash == 0 else float(
    (port["t"] * principal_cash).sum() / total_principal_cash / 12.0
)

# Pool Life (includes defaults as balance runoff; NOT a cash metric)
pool_runoff = port["SchedPrin"] + port["Prepay"] + port["DefaultPrin"]
PoolLife_years = 0.0 if pool_runoff.sum() == 0 else float(
    (port["t"] * pool_runoff).sum() / pool_runoff.sum() / 12.0
)

NPV = float(port["PV"].sum())
EndBal_last = float(port.loc[port["t"] == port["t"].max(), "End_Bal"].values[0]) if not port.empty else 0.0
CumLoss = float(port["CumLoss"].iloc[-1]) if not port.empty else 0.0

# ---------- Waterfall-ready ledger (dùng cho tab Tables) ----------
ledger = pd.DataFrame({
    "t": port["t"],
    "Beg_Bal": port["Beg_Bal"],
    "Gross_Interest": port["Interest"],
    "Servicing_Fees": port["Fees"],
    "Net_Interest_Available": port["Interest"] - port["Fees"],
    "Scheduled_Principal": port["SchedPrin"],
    "Prepayments": port["Prepay"],
    "Principal_Collections": port["SchedPrin"] + port["Prepay"],
    "Defaults_ChargeOffs": port["DefaultPrin"],
    "Recoveries": port["Recoveries"],
    "Net_Principal_Available": (port["SchedPrin"] + port["Prepay"]) - port["DefaultPrin"] + port["Recoveries"],
    "End_Bal": port["End_Bal"],
    "Cashflow": port["Cashflow"],
    "PV": port["PV"],
    "CumLoss": port["CumLoss"],
})

# ---------- Theme & helpers ----------
PRIMARY = "#2457F5"   # cobalt
ACCENT  = "#12B886"   # teal
DANGER  = "#EF4444"

st.markdown(f"""
<style>
.stApp {{
  background: linear-gradient(180deg,#0b1220 0%,#0b1220 60%,#101826 100%);
  color:#e2e8f0;
}}
h1,h2,h3,h4 {{ color:#e6ecff; }}
div[data-testid="stSidebar"] {{ background:#0f172a; color:#e2e8f0; }}
/* metric cards */
div[data-testid="stMetric"] > div {{
  background:#0f172a; border:1px solid #1f2a44; padding:14px 18px; border-radius:14px;
}}
div[data-testid="stMetricLabel"] {{ color:#cbd5e1; }}
div[data-testid="stMetricValue"] {{ color:#e6f0ff; font-weight:700; }}
/* expander */
summary {{ background:#0f172a; border:1px solid #1f2a44; padding:10px 12px; border-radius:10px; }}
details[open] > summary {{ border-bottom:1px solid #1f2a44; }}
/* tables */
thead tr th {{ background:#0f172a !important; color:#cdd6f4 !important; }}
</style>
""", unsafe_allow_html=True)

def fmt_compact_money(x: float, symbol: str = "€") -> str:
    sign = "-" if x < 0 else ""
    n = abs(x)
    if n >= 1e9:  val = f"{n/1e9:.2f}bn"
    elif n >= 1e6: val = f"{n/1e6:.2f}m"
    elif n >= 1e3: val = f"{n/1e3:.0f}k"
    else:         val = f"{n:,.0f}"
    return f"{sign}{symbol}{val}"

def apply_fig_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6ecff"),
    )
    return fig

# Pre-format KPI strings
NPV_str     = fmt_compact_money(NPV)
CumLoss_str = fmt_compact_money(CumLoss)
EndBal_str  = fmt_compact_money(EndBal_last)
WAL_str     = f"{WAL_years:.2f}"
Pool_str    = f"{PoolLife_years:.2f}"

# ---------- TOP-LEVEL NAV ----------
tab_overview, tab_cash, tab_wf, tab_draw, tab_tables = st.tabs(
    ["📊 Overview", "💵 Portfolio Cashflows", "🧱 Waterfalls", "💳 Drawdowns", "📑 Tables & Export"]
)

# =========================
# Tab 1: OVERVIEW
# =========================
with tab_overview:
    st.markdown("### Key performance")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1: st.metric("NPV", NPV_str)
    with r1c2: st.metric("WAL (cash, yrs)", WAL_str)
    with r1c3: st.metric("Pool Life (yrs)", Pool_str)

    r2c1, r2c2 = st.columns(2)
    with r2c1: st.metric("Cumulative Loss", CumLoss_str)
    with r2c2: st.metric(f"End Balance (m{months})", EndBal_str)
    st.markdown("<div style='margin-top:-12px'></div>", unsafe_allow_html=True)

    # Assumptions snapshot (ẩn/hiện)
    with st.expander("Assumptions snapshot (optional)", expanded=False):
        st.json({
            "pd_unit": pd_unit,
            "CPR_annual": CPR_annual,
            "SMM_base": SMM_base,
            "CPR_multiplier": CPR_multiplier,
            "SMM_eff": SMM_eff,
            "servicing_bps": servicing_bps,
            "fee_monthly": (servicing_bps/10000)/12,
            "discount_annual": disc_rate,
            "discount_monthly": d_m,
            "recovery_lag_m": recovery_lag_m,
            "PD_multiplier": PD_multiplier,
            "LGD_shift": LGD_shift,
            "months": months,
            "scenario": scenario,
            "include_recoveries_in_wal": include_recoveries_in_wal,
        })

# =========================
# Tab 2: PORTFOLIO CASHFLOWS
# =========================
with tab_cash:
    st.subheader("Monthly cashflow components (stacked)")
    comp = pd.DataFrame({
        "t": port["t"],
        "Gross Interest": port["Interest"],
        "Scheduled Principal": port["SchedPrin"],
        "Prepayments": port["Prepay"],
        "Recoveries": port["Recoveries"],
        "Servicing Fees": -port["Fees"],
    })
    stack_fig = go.Figure()
    for col in ["Gross Interest","Scheduled Principal","Prepayments","Recoveries","Servicing Fees"]:
        stack_fig.add_trace(go.Bar(x=comp["t"], y=comp[col], name=col))
    stack_fig.update_layout(barmode="relative", xaxis_title="Month", yaxis_title="Amount",
                            title="Monthly Cashflow Components")
    st.plotly_chart(apply_fig_theme(stack_fig), use_container_width=True)

    st.markdown("### Ending balance & cumulatives")
    bal_fig = go.Figure()
    bal_fig.add_trace(go.Scatter(x=port["t"], y=port["End_Bal"], mode="lines", name="Ending Balance"))
    bal_fig.add_trace(go.Scatter(x=port["t"], y=port["Loss"].cumsum(), mode="lines", name="Cumulative Loss"))
    bal_fig.add_trace(go.Scatter(x=port["t"], y=port["Recoveries"].cumsum(), mode="lines", name="Cumulative Recoveries"))
    bal_fig.update_layout(xaxis_title="Month", yaxis_title="Amount",
                          title="Ending Balance, Cum Loss, Cum Recoveries")
    st.plotly_chart(apply_fig_theme(bal_fig), use_container_width=True)

    st.markdown("### Rate trends")
    rate_df = pd.DataFrame({
        "t": port["t"],
        "Monthly Yield (≈ Int/BegBal)": np.where(port["Beg_Bal"]>0, port["Interest"]/port["Beg_Bal"], 0.0)*100,
        "Charge-off Rate (Default/BegBal)": np.where(port["Beg_Bal"]>0, port["DefaultPrin"]/port["Beg_Bal"], 0.0)*100,
        "Prepay Rate (Prepay/BegBal)": np.where(port["Beg_Bal"]>0, port["Prepay"]/port["Beg_Bal"], 0.0)*100,
    })
    rate_fig = go.Figure()
    for col in ["Monthly Yield (≈ Int/BegBal)","Charge-off Rate (Default/BegBal)","Prepay Rate (Prepay/BegBal)"]:
        rate_fig.add_trace(go.Scatter(x=rate_df["t"], y=rate_df[col], mode="lines", name=col))
    rate_fig.update_layout(xaxis_title="Month", yaxis_title="Rate (%)", title="Portfolio Rates Over Time")
    st.plotly_chart(apply_fig_theme(rate_fig), use_container_width=True)

# =========================
# Tab 3: WATERFALLS
# =========================
with tab_wf:
    st.markdown("### Select a month to visualize")
    sel_month = st.slider("Month", 1, int(port["t"].max()), 1, key="wf_sel")
    row = port.loc[port["t"] == sel_month].iloc[0]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Cashflow waterfall**")
        cf_fig = go.Figure(go.Waterfall(
            name="cashflow", orientation="v",
            measure=["relative","relative","relative","relative","relative","total"],
            x=["Gross Interest","Servicing Fees","Scheduled Principal","Prepayments","Recoveries","Net Cashflow"],
            y=[row["Interest"], -row["Fees"], row["SchedPrin"], row["Prepay"], row["Recoveries"], 0],
            text=[f"{row['Interest']:,.0f}", f"{-row['Fees']:,.0f}",
                  f"{row['SchedPrin']:,.0f}", f"{row['Prepay']:,.0f}",
                  f"{row['Recoveries']:,.0f}", f"{row['Cashflow']:,.0f}"],
            textposition="outside",
        ))
        cf_fig.update_layout(title=f"Month {sel_month}: Cashflow Breakdown", showlegend=False, waterfallgap=0.2)
        st.plotly_chart(apply_fig_theme(cf_fig), use_container_width=True)

    with c2:
        st.markdown("**Principal roll-forward**")
        pr_fig = go.Figure(go.Waterfall(
            name="principal", orientation="v",
            measure=["absolute","relative","relative","relative","total"],
            x=["Beginning Balance","Scheduled Principal","Prepayments","Defaults/Charge-offs","Ending Balance"],
            y=[row["Beg_Bal"], -row["SchedPrin"], -row["Prepay"], -row["DefaultPrin"], 0],
            text=[f"{row['Beg_Bal']:,.0f}", f"{-row['SchedPrin']:,.0f}",
                  f"{-row['Prepay']:,.0f}", f"{-row['DefaultPrin']:,.0f}", f"{row['End_Bal']:,.0f}"],
            textposition="outside",
        ))
        pr_fig.update_layout(title=f"Month {sel_month}: Principal Roll-Forward", showlegend=False, waterfallgap=0.2)
        st.plotly_chart(apply_fig_theme(pr_fig), use_container_width=True)

# =========================
# Tab 3: DRAWDOWNS / REVOLVER
# =========================
with tab_draw:
    st.markdown("### Drawdowns / Revolver (illustrative)")
    with st.expander("Parameters", expanded=False):
        enable_draw = st.checkbox("Enable drawdown view", value=True)
        reinv_months = st.number_input("Reinvestment period (months)",
                                       min_value=0, max_value=int(months),
                                       value=min(18, int(months)), step=1)
        fac_rate_annual = st.number_input("Facility interest (annual)",
                                          value=0.06, min_value=0.0, step=0.01, format="%.4f")
        fac_limit = st.number_input("Facility limit (0 = unlimited)", value=0.0, min_value=0.0, step=10000.0,
                                    help="Set 0 for no limit")
        link_drawdown_to_scenario = st.checkbox(
            "Auto-stress with scenario", value=True,
            help="Facility rate / reinvestment window (and optional cap) follow Base/Downside/Upside."
        )

    if enable_draw:
        # Make drawdown params follow the scenario
        if link_drawdown_to_scenario:
            opening_pool = float(port["Beg_Bal"].iloc[0]) if not port.empty else 0.0
            if scenario == "Downside":
                fac_rate_annual = max(fac_rate_annual, 0.075)
                reinv_months    = min(reinv_months, 9)
                if fac_limit == 0.0:
                    fac_limit = 0.85 * opening_pool
            elif scenario == "Upside":
                fac_rate_annual = min(fac_rate_annual, 0.050)
                reinv_months    = min(reinv_months, 24)
            else:
                fac_rate_annual = max(fac_rate_annual, 0.060)
                reinv_months    = min(reinv_months, 18)

        # Purchases = run-off to keep pool flat during reinvestment
        required_purchases = port["Beg_Bal"] - port["End_Bal"]
        npa = (port["SchedPrin"] + port["Prepay"]) - port["DefaultPrin"] + port["Recoveries"]
        net_int_avail = port["Interest"] - port["Fees"]
        fac_rate_m = fac_rate_annual / 12.0

        rows, fac_beg, cum_draw, cum_repay = [], 0.0, 0.0, 0.0
        for i in range(len(port)):
            t = int(port.loc[i, "t"])
            req = float(required_purchases.iloc[i])
            npa_t = float(npa.iloc[i])
            nia_t = float(net_int_avail.iloc[i])

            fac_interest = fac_beg * fac_rate_m
            shortfall = req - npa_t
            draw = repay = 0.0

            if t <= reinv_months:
                if shortfall > 1e-9:
                    draw = shortfall
                    if fac_limit > 0.0:
                        draw = max(0.0, min(draw, fac_limit - fac_beg))
                elif shortfall < -1e-9:
                    repay = min(-shortfall, fac_beg)
            else:
                repay = min(max(0.0, npa_t), fac_beg)

            interest_shortfall = max(0.0, fac_interest - nia_t)

            fac_end = fac_beg + draw - repay
            cum_draw += draw; cum_repay += repay

            rows.append({
                "t": t, "Beg_Facility": fac_beg, "Required_Purchases": req, "NPA": npa_t,
                "Shortfall(+)/Excess(-)": shortfall, "Draw": draw, "Repay": repay, "End_Facility": fac_end,
                "Facility_Interest_Paid": fac_interest, "Net_Interest_Available": nia_t,
                "Interest_Shortfall": interest_shortfall, "Cum_Draws": cum_draw, "Cum_Repays": cum_repay,
            })
            fac_beg = fac_end

        drawdf = pd.DataFrame(rows)

        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("Max Facility Balance", f"{drawdf['End_Facility'].max():,.0f}")
        kc2.metric("Final Facility Balance", f"{drawdf['End_Facility'].iloc[-1]:,.0f}")
        kc3.metric("Total Facility Interest Paid", f"{drawdf['Facility_Interest_Paid'].sum():,.0f}")

        # Charts
        fac_line = go.Figure(go.Scatter(x=drawdf["t"], y=drawdf["End_Facility"],
                                        mode="lines+markers", name="Facility Balance"))
        fac_line.update_layout(title="Facility Balance over time", xaxis_title="Month", yaxis_title="Balance")
        st.plotly_chart(apply_fig_theme(fac_line), use_container_width=True)

        dr = go.Figure()
        dr.add_trace(go.Bar(x=drawdf["t"], y=drawdf["Draw"], name="Draws"))
        dr.add_trace(go.Bar(x=drawdf["t"], y=-drawdf["Repay"], name="Repayments"))
        dr.update_layout(barmode="relative", title="Monthly Draws and Repayments",
                         xaxis_title="Month", yaxis_title="Amount")
        st.plotly_chart(apply_fig_theme(dr), use_container_width=True)

        cov = go.Figure()
        cov.add_trace(go.Bar(x=drawdf["t"], y=drawdf["Net_Interest_Available"], name="Net Interest Available"))
        cov.add_trace(go.Bar(x=drawdf["t"], y=-drawdf["Facility_Interest_Paid"], name="Facility Interest"))
        cov.add_trace(go.Bar(x=drawdf["t"], y=-drawdf["Interest_Shortfall"], name="Interest Shortfall"))
        cov.update_layout(barmode="relative", title="Interest Coverage",
                          xaxis_title="Month", yaxis_title="Amount")
        st.plotly_chart(apply_fig_theme(cov), use_container_width=True)

        with st.expander("Drawdown schedule (table)"):
            st.dataframe(drawdf.style.format("{:,.0f}"), use_container_width=True)
            st.download_button(
                "Download drawdown schedule CSV",
                drawdf.to_csv(index=False).encode("utf-8"),
                file_name="drawdown_schedule.csv",
                mime="text/csv",
            )

# =========================
# Tab 4: TABLES & EXPORT
# =========================
with tab_tables:
    with st.expander("Portfolio cashflows (monthly)", expanded=False):
        st.dataframe(port.style.format("{:,.0f}"), use_container_width=True)

    with st.expander("Waterfall-ready ledger", expanded=False):
        st.dataframe(ledger.style.format("{:,.0f}"), use_container_width=True)

    with st.expander("Loan-level cashflows", expanded=False):
        st.dataframe(cf.style.format("{:,.0f}"), use_container_width=True)

    # One-click Excel + CSVs
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        port.to_excel(xw, index=False, sheet_name="portfolio")
        ledger.to_excel(xw, index=False, sheet_name="ledger")
        cf.to_excel(xw, index=False, sheet_name="loan_level")
    st.download_button(
        "⬇️ Download XLSX (portfolio, ledger, loan-level)",
        data=buf.getvalue(),
        file_name="cashflows.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    c_dl1, c_dl2, c_dl3 = st.columns(3)
    with c_dl1:
        st.download_button("CSV — portfolio", port.to_csv(index=False).encode("utf-8"),
                           file_name="portfolio_cashflows.csv", mime="text/csv", use_container_width=True)
    with c_dl2:
        st.download_button("CSV — loan-level", cf.to_csv(index=False).encode("utf-8"),
                           file_name="loanlevel_cashflows.csv", mime="text/csv", use_container_width=True)
    with c_dl3:
        st.download_button("CSV — ledger", ledger.to_csv(index=False).encode("utf-8"),
                           file_name="waterfall_feed.csv", mime="text/csv", use_container_width=True)





