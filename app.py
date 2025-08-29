"""
Created on Wed Aug 27 23:52:46 2025

@author: LucyVu
"""
import pandas as pd
import numpy as np
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
    loans = read_loan_tape(file, sheet_name=sheet_name)
elif use_sample:
    if os.path.exists(SAMPLE_PATH):
        loans = read_loan_tape(SAMPLE_PATH, sheet_name=sheet_name)
    else:
        csv_path = os.path.splitext(SAMPLE_PATH)[0] + ".csv"
        if os.path.exists(csv_path):
            loans = pd.read_csv(csv_path)

if loans is None:
    st.info("Upload a loan tape or tick 'Use bundled sample file' to run the model.")
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

st.subheader("Loan tape preview")
st.dataframe(loans.head(), use_container_width=True)

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
# WAL (cash) — exclude defaults; optionally include recoveries if you tick the box
principal_cash = port["SchedPrin"] + port["Prepay"]
if include_recoveries_in_wal:
    principal_cash = principal_cash + port["Recoveries"]

total_principal_cash = principal_cash.sum()
WAL_years = 0.0 if total_principal_cash == 0 else \
    float((port["t"] * principal_cash).sum() / total_principal_cash / 12.0)

# Pool Life (includes defaults as balance runoff; NOT a cash metric)
pool_runoff = port["SchedPrin"] + port["Prepay"] + port["DefaultPrin"]
PoolLife_years = 0.0 if pool_runoff.sum() == 0 else \
    float((port["t"] * pool_runoff).sum() / pool_runoff.sum() / 12.0)

NPV = float(port["PV"].sum())
EndBal_last = float(port.loc[port["t"] == port["t"].max(), "End_Bal"].values[0]) if not port.empty else 0.0
CumLoss = float(port["CumLoss"].iloc[-1]) if not port.empty else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("NPV", f"{NPV:,.0f}")
k2.metric("WAL (cash, yrs)", f"{WAL_years:,.2f}")
k3.metric("Pool Life (yrs)", f"{PoolLife_years:,.2f}")
k4.metric("Cumulative Loss", f"{CumLoss:,.0f}")
k5.metric(f"End Balance (m{months})", f"{EndBal_last:,.0f}")

# ---------- Waterfall-ready ledger ----------
ledger = pd.DataFrame({
    "t": port["t"],
    "Beg_Bal": port["Beg_Bal"],
    # Interest side
    "Gross_Interest": port["Interest"],
    "Servicing_Fees": port["Fees"],
    "Net_Interest_Available": port["Interest"] - port["Fees"],
    # Principal side
    "Scheduled_Principal": port["SchedPrin"],
    "Prepayments": port["Prepay"],
    "Principal_Collections": port["SchedPrin"] + port["Prepay"],
    "Defaults_ChargeOffs": port["DefaultPrin"],
    "Recoveries": port["Recoveries"],
    "Net_Principal_Available": (port["SchedPrin"] + port["Prepay"]) - port["DefaultPrin"] + port["Recoveries"],
    # Roll-forward & valuation
    "End_Bal": port["End_Bal"],
    "Cashflow": port["Cashflow"],
    "PV": port["PV"],
    "CumLoss": port["CumLoss"],
})

st.subheader("Portfolio cashflows (monthly)")
st.dataframe(port.style.format("{:,.0f}"), use_container_width=True)

st.subheader("Waterfall-ready ledger")
st.dataframe(ledger.style.format("{:,.0f}"), use_container_width=True)

# ---------- Single-month Cashflow Waterfall ----------
st.subheader("Waterfall — Cashflow (select month)")

if not port.empty:
    sel_month = st.slider("Select month for cashflow waterfall", 1, int(port["t"].max()), 1)
    row = port.loc[port["t"] == sel_month].iloc[0]

    cf_fig = go.Figure(go.Waterfall(
        name="cashflow",
        orientation="v",
        measure=["relative","relative","relative","relative","relative","total"],
        x=["Gross Interest", "Servicing Fees", "Scheduled Principal", "Prepayments", "Recoveries", "Net Cashflow"],
        y=[row["Interest"], -row["Fees"], row["SchedPrin"], row["Prepay"], row["Recoveries"], 0],
        text=[f"{row['Interest']:,.0f}", f"{-row['Fees']:,.0f}", f"{row['SchedPrin']:,.0f}",
              f"{row['Prepay']:,.0f}", f"{row['Recoveries']:,.0f}", f"{row['Cashflow']:,.0f}"],
        textposition="outside"
    ))
    cf_fig.update_layout(
        showlegend=False,
        waterfallgap=0.2,
        title=f"Month {sel_month}: Cashflow Breakdown"
    )
    st.plotly_chart(cf_fig, use_container_width=True)

# ---------- Single-month Principal Roll-Forward Waterfall ----------
st.subheader("Waterfall — Principal Roll-Forward (select month)")

if not port.empty:
    sel_month2 = st.slider("Select month for balance waterfall", 1, int(port["t"].max()), 1, key="bal_wf")
    row2 = port.loc[port["t"] == sel_month2].iloc[0]

    pr_fig = go.Figure(go.Waterfall(
        name="principal",
        orientation="v",
        measure=["absolute","relative","relative","relative","total"],
        x=["Beginning Balance", "Scheduled Principal", "Prepayments", "Defaults/Charge-offs", "Ending Balance"],
        y=[row2["Beg_Bal"], -row2["SchedPrin"], -row2["Prepay"], -row2["DefaultPrin"], 0],
        text=[f"{row2['Beg_Bal']:,.0f}", f"{-row2['SchedPrin']:,.0f}", f"{-row2['Prepay']:,.0f}",
              f"{-row2['DefaultPrin']:,.0f}", f"{row2['End_Bal']:,.0f}"],
        textposition="outside"
    ))
    pr_fig.update_layout(
        showlegend=False,
        waterfallgap=0.2,
        title=f"Month {sel_month2}: Principal Roll-Forward"
    )
    st.plotly_chart(pr_fig, use_container_width=True)

# ---------- Portfolio cashflow components (stacked bars) ----------
st.subheader("Portfolio – Monthly Cashflow Components (stacked)")

comp = pd.DataFrame({
    "t": port["t"],
    "Gross Interest": port["Interest"],
    "Scheduled Principal": port["SchedPrin"],
    "Prepayments": port["Prepay"],
    "Recoveries": port["Recoveries"],
    "Servicing Fees": -port["Fees"],   # negative to show below axis
})

stack_fig = go.Figure()
for col in ["Gross Interest","Scheduled Principal","Prepayments","Recoveries","Servicing Fees"]:
    stack_fig.add_trace(go.Bar(x=comp["t"], y=comp[col], name=col))

stack_fig.update_layout(
    barmode="relative",  # stacks positives and negatives around zero
    xaxis_title="Month",
    yaxis_title="Amount",
    title="Monthly Cashflow Components",
)
st.plotly_chart(stack_fig, use_container_width=True)

# ---------- Downloads ----------
st.download_button(
    "Download portfolio CSV",
    port.to_csv(index=False).encode("utf-8"),
    file_name="portfolio_cashflows.csv",
    mime="text/csv",
)
st.download_button(
    "Download loan-level CSV (Base)",
    cf.to_csv(index=False).encode("utf-8"),
    file_name="loanlevel_cashflows.csv",
    mime="text/csv",
)
st.download_button(
    "Download waterfall ledger CSV",
    ledger.to_csv(index=False).encode("utf-8"),
    file_name="waterfall_feed.csv",
    mime="text/csv",
)

# ---------- Assumption log ----------
st.subheader("Assumption log")
st.json({
    "pd_unit": pd_unit,
    "CPR_annual": CPR_annual,
    "SMM_base": SMM_base,
    "CPR_multiplier": CPR_multiplier,
    "SMM_eff": SMM_eff,
    "servicing_bps": servicing_bps,
    "fee_monthly": fee_m,
    "discount_annual": disc_rate,
    "discount_monthly": d_m,
    "recovery_lag_m": recovery_lag_m,
    "PD_multiplier": PD_multiplier,
    "LGD_shift": LGD_shift,
    "months": months,
    "scenario": scenario,
    "include_recoveries_in_wal": include_recoveries_in_wal,
})

