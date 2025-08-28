"""
Created on Wed Aug 27 23:52:46 2025

@author: LucyVu
"""
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Consumer Loans – 60M Expected CF", layout="wide")
st.title("German Consumer Loans — 60-Month Expected Cashflows")

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

# ---------- Sidebar: data & assumptions ----------
st.sidebar.header("Data source")
file = st.sidebar.file_uploader("Upload file", type=["xlsx", "csv"])
use_sample = st.sidebar.checkbox("Use bundled sample file: 'loanslite for app.xlsx'", value=True)
SAMPLE_PATH = "loanslite for app.xlsx"

st.sidebar.header("Assumptions")
CPR_annual = st.sidebar.number_input("CPR (annual)", value=0.08, min_value=0.0, step=0.01, format="%.4f")
SMM_base = cpr_to_smm(CPR_annual)
servicing_bps = st.sidebar.number_input("Servicing fee (bps/year)", value=50, min_value=0, step=5)
disc_rate = st.sidebar.number_input("Discount rate (annual)", value=0.07, min_value=0.0, step=0.01, format="%.4f")
recovery_lag_m = st.sidebar.number_input("Recovery lag (months)", value=12, min_value=0, step=1)  # 12 as a reasonable default
months = st.sidebar.slider("Projection horizon (months)", 1, 60, 60)

st.sidebar.header("Scenarios & Overrides")
scenario = st.sidebar.selectbox("Scenario", ["Base", "Downside", "Upside"])
PD_multiplier = st.sidebar.number_input("PD multiplier (k)", value=1.0, step=0.1, format="%.2f")
LGD_shift = st.sidebar.number_input("LGD shift (±Δ)", value=0.0, step=0.05, format="%.2f")
CPR_multiplier = st.sidebar.number_input("CPR multiplier (m)", value=1.0, step=0.1, format="%.2f")

# Apply preset scenario adjustments (tweak as you like)
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
    if file.name.lower().endswith(".csv"):
        loans = pd.read_csv(file)
    else:
        loans = pd.read_excel(file, sheet_name="loan_tape")
elif use_sample:
    try:
        loans = pd.read_excel(SAMPLE_PATH, sheet_name="loan_tape")
    except FileNotFoundError:
        st.error(
            f"Sample file not found at '{SAMPLE_PATH}'. "
            "Upload a loan tape or place the sample alongside app.py."
        )
        st.stop()

if loans is None:
    st.info("Upload a loan tape (or tick 'Use bundled sample file') to run the model.")
    st.stop()

# Check required schema
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

        pd_m = float(r["pd"]) * PD_multiplier
        pd_m = min(max(pd_m, 0.0), 1.0)
        lgd = min(max(float(r["lgd"]) + LGD_shift, 0.0), 1.0)

        # bucket for delayed recoveries
        recs = {}
        for t in range(1, horizon_months + 1):
            if bal < 1e-9:
                # Only delayed recoveries may arrive after balance goes to zero
                cash = recs.get(t, 0.0)
                pv = cash / ((1 + d_m) ** t)
                rows.append([
                    t, loan_id, 0.0, 0.0, 0.0, 0.0, 0.0,
                    recs.get(t, 0.0), 0.0, 0.0, cash, pv, 0.0
                ])
                continue

            # 1) Interest
            interest = bal * r_m

            # 2) Scheduled principal
            sched_prin = max(0.0, min(bal, pmt - interest))

            # 3) Prepayment
            prepay_base = max(0.0, bal - sched_prin)
            prepay = prepay_base * SMM_eff

            # 4) Default
            default_base = max(0.0, prepay_base - prepay)
            def_prin = default_base * pd_m

            # 5) Loss & Recovery
            loss = def_prin * lgd
            rec_amount = def_prin * (1 - lgd)
            lag_month = int(recovery_lag_m)
            recs[t + lag_month] = recs.get(t + lag_month, 0.0) + rec_amount

            # 6) Fees (on beginning balance)
            fees = bal * fee_m

            # 7) End balance
            end_bal = max(0.0, bal - sched_prin - prepay - def_prin)

            # 8) Cashflow & PV
            cash = interest + sched_prin + prepay + recs.get(t, 0.0) - fees
            pv = cash / ((1 + d_m) ** t)

            rows.append([
                t, loan_id, bal, interest, sched_prin, prepay, def_prin,
                recs.get(t, 0.0), fees, end_bal, cash, pv, loss
            ])

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

# KPIs
principal_flows = port["SchedPrin"] + port["Prepay"] + port["DefaultPrin"]
total_principal = principal_flows.sum()
WAL_years = 0.0 if total_principal == 0 else float((port["t"] * principal_flows).sum() / total_principal / 12.0)
NPV = float(port["PV"].sum())
EndBal_m60 = float(port.loc[port["t"] == port["t"].max(), "End_Bal"].values[0]) if not port.empty else 0.0
CumLoss = float(port["CumLoss"].iloc[-1]) if not port.empty else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("NPV", f"{NPV:,.0f}")
k2.metric("WAL (yrs)", f"{WAL_years:,.2f}")
k3.metric("Cumulative Loss", f"{CumLoss:,.0f}")
k4.metric("End Balance (m60)", f"{EndBal_m60:,.0f}")

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
})

