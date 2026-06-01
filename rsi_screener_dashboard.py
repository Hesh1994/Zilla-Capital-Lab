"""
rsi_screener_dashboard.py
=========================
Streamlit RSI Screener Dashboard

Sidebar-driven config → fetches tickers (Wikipedia / curated lists) →
downloads price history via yfinance → calculates Wilder RSI →
displays oversold / overbought tables with colour-coded RSI columns.

Run:
    streamlit run rsi_screener_dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import requests
from datetime import datetime, timedelta
from io import StringIO

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RSI Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 RSI Stock Screener")
st.caption("Fetches tickers, downloads price history via yfinance, and ranks by Wilder RSI.")

# ---------------------------------------------------------------------------
# Ticker sources
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

TICKER_SOURCES = {
    "US — S&P 500":   {"country": "US", "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "table_idx": 0, "ticker_col": "Symbol",      "name_col": "Security",      "suffix": ""},
    "GB — FTSE 100":  {"country": "GB", "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",              "table_idx": 4, "ticker_col": "Ticker",      "name_col": "Company",       "suffix": ".L"},
    "DE — DAX 40":    {"country": "DE", "url": "https://en.wikipedia.org/wiki/DAX",                         "table_idx": 4, "ticker_col": "Ticker",      "name_col": "Company",       "suffix": ".DE"},
    "FR — CAC 40":    {"country": "FR", "url": "https://en.wikipedia.org/wiki/CAC_40",                      "table_idx": 4, "ticker_col": "Ticker",      "name_col": "Company",       "suffix": ".PA"},
    "CA — TSX 60":    {"country": "CA", "url": "https://en.wikipedia.org/wiki/S%26P/TSX_60",                "table_idx": 1, "ticker_col": "Symbol",      "name_col": "Company",       "suffix": ".TO"},
    "AU — ASX 200":   {"country": "AU", "url": "https://en.wikipedia.org/wiki/S%26P/ASX_200",               "table_idx": 2, "ticker_col": "Code",        "name_col": "Company",       "suffix": ".AX"},
    "IN — NIFTY 50":  {"country": "IN", "url": "https://en.wikipedia.org/wiki/NIFTY_50",                    "table_idx": 1, "ticker_col": "Symbol",      "name_col": "Company name",  "suffix": ".NS"},
    "SA — Tadawul":   {"country": "SA", "url": None, "table_idx": None, "ticker_col": None, "name_col": None, "suffix": ""},
}

SA_TICKERS = [
    ("2222.SR","Saudi Aramco"),    ("1180.SR","Al Rajhi Bank"),   ("1120.SR","Al Jazira Bank"),
    ("2010.SR","SABIC"),           ("1150.SR","Alinma Bank"),     ("7010.SR","STC"),
    ("4200.SR","Saudi Electricity"),("2350.SR","Saudi Kayan"),    ("1030.SR","SAMBA"),
    ("1050.SR","Banque Saudi Fransi"),("2380.SR","Petro Rabigh"), ("3010.SR","SABIC Agri-Nutrients"),
    ("2060.SR","SIPCHEM"),         ("4240.SR","Saudi Telecom"),   ("1140.SR","Al Rajhi Takaful"),
]

INTERVAL_MAP = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
ALL_INTERVALS = ["Daily", "Weekly", "Monthly"]

# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    market_label = st.selectbox("Market", list(TICKER_SOURCES.keys()), index=0)
    rsi_window   = st.number_input("RSI Window", min_value=2, max_value=50, value=14, step=1)
    interval     = st.selectbox("Price Interval", ALL_INTERVALS, index=0)
    max_tickers  = st.number_input("Max Tickers", min_value=5, max_value=500, value=50, step=5)

    st.divider()
    st.subheader("Thresholds")
    oversold_thresh   = st.number_input("Oversold  (RSI <)",  min_value=1,  max_value=99, value=30, step=1)
    overbought_thresh = st.number_input("Overbought (RSI >)", min_value=1,  max_value=99, value=70, step=1)

    run = st.button("Run Screener", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tickers(market_label: str, max_tickers: int) -> pd.DataFrame:
    src = TICKER_SOURCES[market_label]
    if src["country"] == "SA":
        df = pd.DataFrame(SA_TICKERS, columns=["ticker", "company_name"])
        return df.head(max_tickers)
    html   = requests.get(src["url"], headers=HEADERS, timeout=30).text
    tables = pd.read_html(StringIO(html))
    table  = tables[src["table_idx"]]
    df = table[[src["ticker_col"], src["name_col"]]].copy()
    df.columns = ["ticker", "company_name"]
    df = df.dropna(subset=["ticker"]).reset_index(drop=True)
    df["ticker"] = df["ticker"].str.strip() + src["suffix"]
    return df.head(max_tickers)


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_price_history(ticker: str, interval: str, rsi_window: int):
    yf_interval = INTERVAL_MAP[interval]
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=start, end=end, interval=yf_interval,
                         auto_adjust=True, progress=False)
    except Exception as e:
        return None, str(e)
    if df is None or df.empty:
        return None, "empty"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if len(df) < rsi_window + 1:
        return None, f"only {len(df)} bars"
    return df, None


def calculate_rsi_wilder(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    n = len(close)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[window] = gain.iloc[1:window + 1].mean()
    avg_loss[window] = loss.iloc[1:window + 1].mean()
    for i in range(window + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain.iloc[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss.iloc[i]) / window
    ag = pd.Series(avg_gain, index=close.index)
    al = pd.Series(avg_loss, index=close.index)
    rs  = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def get_latest_rsi(ticker: str, interval: str, rsi_window: int):
    df, err = fetch_price_history(ticker, interval, rsi_window)
    if df is None:
        return None
    try:
        series = calculate_rsi_wilder(df["close"], rsi_window)
        return round(float(series.dropna().iloc[-1]), 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RSI colour helper for st.dataframe
# ---------------------------------------------------------------------------

def rsi_color(val, criteria):
    if pd.isna(val):
        return ""
    if criteria == "oversold":
        # green when deeply oversold, red when near 50
        if val < 20:   return "background-color:#1a7f37; color:white"
        elif val < 30: return "background-color:#4caf50; color:white"
        elif val < 40: return "background-color:#a5d6a7"
        else:          return ""
    else:
        if val > 80:   return "background-color:#c62828; color:white"
        elif val > 70: return "background-color:#ef5350; color:white"
        elif val > 60: return "background-color:#ef9a9a"
        else:          return ""


# ---------------------------------------------------------------------------
# Main run logic
# ---------------------------------------------------------------------------

if not run:
    st.info("Configure the screener in the sidebar, then click **Run Screener**.")
    st.stop()

# ── Step 1: fetch tickers ────────────────────────────────────────────────────
with st.spinner("Loading ticker list..."):
    try:
        tickers_df = fetch_tickers(market_label, int(max_tickers))
    except Exception as e:
        st.error(f"Failed to load tickers: {e}")
        st.stop()

st.success(f"Loaded **{len(tickers_df)}** tickers for **{market_label}**")

# ── Step 2: fetch prices & compute RSI ──────────────────────────────────────
progress_bar = st.progress(0, text="Fetching price data and computing RSI...")
status_text  = st.empty()

rsi_latest  = {}
price_cache = {}
failed      = {}
total       = len(tickers_df)

for idx, row in tickers_df.iterrows():
    ticker = row["ticker"]
    status_text.text(f"Processing {ticker} ({idx + 1}/{total})")
    progress_bar.progress((idx + 1) / total)

    df, err = fetch_price_history(ticker, interval, int(rsi_window))
    if df is not None:
        try:
            series = calculate_rsi_wilder(df["close"], int(rsi_window))
            rsi_latest[ticker]  = round(float(series.dropna().iloc[-1]), 2)
            price_cache[ticker] = (df, series)
        except Exception as e:
            failed[ticker] = str(e)
    else:
        failed[ticker] = err

progress_bar.empty()
status_text.empty()

col1, col2, col3 = st.columns(3)
col1.metric("Tickers processed", len(rsi_latest))
col2.metric("Failed / skipped",  len(failed))
col3.metric("Interval",          interval)

# ── Step 3: filter oversold / overbought ────────────────────────────────────
oversold_dict   = {t: v for t, v in rsi_latest.items() if v < oversold_thresh}
overbought_dict = {t: v for t, v in rsi_latest.items() if v > overbought_thresh}

# ── Step 4: build multi-timeframe rows ──────────────────────────────────────

def build_rows(ticker_dict: dict, label: str) -> pd.DataFrame:
    other_intervals = [iv for iv in ALL_INTERVALS if iv != interval]
    rows = []
    prog = st.progress(0, text=f"Fetching secondary timeframes for {label}...")
    total_t = len(ticker_dict)
    for i, ticker in enumerate(ticker_dict):
        prog.progress((i + 1) / total_t)
        company = tickers_df.loc[tickers_df["ticker"] == ticker, "company_name"].values
        company = company[0] if len(company) else ""
        row = {"Ticker": ticker, "Company": company, f"RSI ({interval})": ticker_dict[ticker]}
        for iv in other_intervals:
            row[f"RSI ({iv})"] = get_latest_rsi(ticker, iv, int(rsi_window))
        rows.append(row)
    prog.empty()
    if not rows:
        return pd.DataFrame()
    cols_order = ["Ticker", "Company", "RSI (Daily)", "RSI (Weekly)", "RSI (Monthly)"]
    df = pd.DataFrame(rows)
    existing = [c for c in cols_order if c in df.columns]
    return df[existing].sort_values(f"RSI ({interval})", ascending=(label == "oversold")).reset_index(drop=True)


# ── Step 5: display results ──────────────────────────────────────────────────
tab_over, tab_overbought, tab_chart = st.tabs(
    [f"Oversold  ({len(oversold_dict)})",
     f"Overbought  ({len(overbought_dict)})",
     "RSI Chart"]
)

rsi_cols = ["RSI (Daily)", "RSI (Weekly)", "RSI (Monthly)"]

with tab_over:
    st.subheader(f"Oversold — RSI < {oversold_thresh}")
    if oversold_dict:
        df_over = build_rows(oversold_dict, "oversold")
        rsi_present = [c for c in rsi_cols if c in df_over.columns]
        styled = df_over.style.map(
            lambda v: rsi_color(v, "oversold"), subset=rsi_present
        ).format({c: "{:.2f}" for c in rsi_present}, na_rep="—")
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info(f"No stocks with RSI < {oversold_thresh} found. Try raising the threshold.")

with tab_overbought:
    st.subheader(f"Overbought — RSI > {overbought_thresh}")
    if overbought_dict:
        df_ob = build_rows(overbought_dict, "overbought")
        rsi_present = [c for c in rsi_cols if c in df_ob.columns]
        styled = df_ob.style.map(
            lambda v: rsi_color(v, "overbought"), subset=rsi_present
        ).format({c: "{:.2f}" for c in rsi_present}, na_rep="—")
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info(f"No stocks with RSI > {overbought_thresh} found. Try lowering the threshold.")

with tab_chart:
    st.subheader("RSI Chart — individual ticker")
    all_processed = list(rsi_latest.keys())
    if all_processed:
        selected = st.selectbox("Select ticker", all_processed)
        if selected in price_cache:
            df_px, rsi_series = price_cache[selected]

            fig = go.Figure()

            # Price candlestick
            fig.add_trace(go.Candlestick(
                x=df_px.index,
                open=df_px["open"], high=df_px["high"],
                low=df_px["low"],   close=df_px["close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ))
            fig.update_layout(
                title=f"{selected} — Price ({interval})",
                xaxis_rangeslider_visible=False,
                height=300,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # RSI line
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=rsi_series.index, y=rsi_series,
                mode="lines", name="RSI", line=dict(color="#7b61ff", width=1.5)
            ))
            fig2.add_hline(y=oversold_thresh,   line_dash="dash", line_color="green",
                           annotation_text=f"Oversold {oversold_thresh}")
            fig2.add_hline(y=overbought_thresh,  line_dash="dash", line_color="red",
                           annotation_text=f"Overbought {overbought_thresh}")
            fig2.add_hline(y=50, line_dash="dot", line_color="gray", line_width=0.8)
            fig2.update_layout(
                title=f"{selected} — RSI ({rsi_window})",
                yaxis=dict(range=[0, 100]),
                height=250,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

            latest_rsi = rsi_latest[selected]
            if latest_rsi < oversold_thresh:
                st.success(f"RSI = **{latest_rsi}** — Oversold signal")
            elif latest_rsi > overbought_thresh:
                st.error(f"RSI = **{latest_rsi}** — Overbought signal")
            else:
                st.info(f"RSI = **{latest_rsi}** — Neutral")
    else:
        st.info("Run the screener first.")

# ── Failed tickers expander ──────────────────────────────────────────────────
if failed:
    with st.expander(f"Failed tickers ({len(failed)})"):
        st.dataframe(
            pd.DataFrame({"Ticker": list(failed.keys()), "Reason": list(failed.values())}),
            hide_index=True, use_container_width=True,
        )
