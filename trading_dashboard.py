# Trading Strategy Backtesting Dashboard - Version 3.0
# Updated: July 20, 2025 - Completely self-contained with manual calculations

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config first
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.success("✅ Using manual technical analysis calculations")

st.title("📈 Trading Strategy Backtesting Dashboard")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("🔧 Configuration")

# Ticker Symbol Input
ticker = st.sidebar.text_input("📊 Ticker Symbol", value="MSFT", help="Enter stock ticker symbol (e.g., MSFT, AAPL, GOOGL)")

# Date Range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("📅 Start Date", value=pd.to_datetime("2010-01-01"))
with col2:
    end_date = st.date_input("📅 End Date", value=pd.to_datetime("2025-07-07"))

# Price Column Selection
price_options = ["Adj Close", "Close", "Open", "High", "Low"]
price_column = st.sidebar.selectbox("💰 Price Column", price_options, index=0)

st.sidebar.markdown("---")
st.sidebar.header("📊 Indicator Parameters")

# RSI Parameters
st.sidebar.subheader("RSI Settings")
rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=200, value=30, step=1)
rsi_mid_period = st.sidebar.slider("RSI Mid Period", min_value=5, max_value=200, value=30, step=1)

# SMA/EMA Parameters
st.sidebar.subheader("Moving Average Settings")
sma_period = st.sidebar.slider("SMA Period", min_value=5, max_value=200, value=30, step=1)
ema_period = st.sidebar.slider("EMA Period", min_value=5, max_value=200, value=15, step=1)
ema_seed_period = st.sidebar.slider("EMA Seed Period", min_value=5, max_value=200, value=15, step=1)

# Strategy Selection
st.sidebar.markdown("---")
st.sidebar.header("🎯 Strategy Selection")
strategy_options = [
    "RSI_ONLY",
    "RSI_50_ONLY", 
    "SMA_ONLY",
    "RSI_WITH_RSI_50",
    "SMA_WITH_RSI",
    "SMA_RSI_50_COMBO",
    "RSI_RSI50_SMA_COMBO",
    "RSI_RSI50_COMBO",
    "RSI_SMA_COMBO",
    "RSI50_SMA_COMBO"
]

selected_strategies = st.sidebar.multiselect(
    "Select Strategies to Backtest",
    strategy_options,
    default=["RSI_ONLY", "RSI_50_ONLY", "SMA_ONLY", "RSI_RSI50_COMBO"]
)

# Analysis Functions
@st.cache_data
def download_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        data = yf.download(tickers=ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False)
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

@st.cache_data
def download_sp500_data(start_date, end_date):
    """Download S&P 500 data for comparison"""
    try:
        sp500_data = yf.download(tickers='^GSPC', start=start_date, end=end_date, interval='1d', auto_adjust=False)
        return sp500_data
    except Exception as e:
        st.error(f"Error downloading S&P 500 data: {e}")
        return None

def calculate_max_drawdown(returns_series):
    """Calculate maximum drawdown from a returns series"""
    if returns_series.empty or len(returns_series) == 0:
        return 0
    
    cumulative = (1 + returns_series).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    
    # Handle cases where drawdown might be empty or all NaN
    if drawdown.empty or drawdown.isna().all():
        return 0
    
    return drawdown.min()

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.02):
    """Calculate Sharpe ratio (assuming 2% risk-free rate)"""
    if returns_series.empty or len(returns_series) == 0:
        return 0
    
    excess_returns = returns_series - risk_free_rate/252  # Daily risk-free rate
    
    # Handle cases where excess_returns might be all NaN
    if excess_returns.isna().all():
        return 0
    
    std_dev = excess_returns.std()
    mean_return = excess_returns.mean()
    
    # Check for invalid values
    if pd.isna(std_dev) or pd.isna(mean_return) or std_dev == 0 or std_dev < 1e-10:
        return 0
    
    return (mean_return * 252) / (std_dev * np.sqrt(252))

def safe_column_access(df, column_candidates):
    """Safely access DataFrame columns, handling both single-level and MultiIndex"""
    if df is None or df.empty:
        return None
    
    # Convert columns to list to avoid Series boolean issues
    available_columns = list(df.columns)
    
    # Try each column candidate
    for col_name in column_candidates:
        try:
            # Direct lookup in the list
            if col_name in available_columns:
                return df[col_name]
        except Exception:
            continue
    
    return None

def calculate_indicators(df, ticker, price_col, rsi_period, rsi_mid_period, sma_period, ema_period, ema_seed_period):
    """Calculate technical indicators"""
    prices = df[price_col, ticker].values
    
    # Calculate EMA manually
    alpha = 2 / (ema_period + 1)
    ema_values = []
    
    for i in range(len(prices)):
        if i == ema_seed_period - 1:
            sma_seed = np.mean(prices[:ema_seed_period])
            ema_values.append(sma_seed)
        elif i >= ema_seed_period:
            current_price = prices[i]
            previous_ema = ema_values[-1]
            new_ema = (current_price * alpha) + (previous_ema * (1 - alpha))
            ema_values.append(new_ema)
        else:
            ema_values.append(np.nan)
    
    df['ema_manual'] = ema_values
    df['sma'] = df[price_col, ticker].rolling(window=sma_period).mean()
    
    # Calculate RSI manually (Relative Strength Index)
    delta = df[price_col, ticker].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['rsi_mid'] = df['rsi'].rolling(window=rsi_mid_period).mean()
    
    # Signals
    df['sma_signal'] = np.where(df['ema_manual'] > df['sma'], 1, 0)
    df['rsi_signal'] = np.where(df['rsi'] > df['rsi_mid'], 1, 0)
    df['rsi_50_signal'] = np.where(df['rsi_mid'] > 50, 1, 0)
    
    df['returns'] = df[price_col, ticker].pct_change()
    df['returns_acc'] = (1 + df['returns'].fillna(0)).cumprod()
    
    # Signal combinations
    df['signal_combo'] = list(zip(df['rsi_signal'], df['rsi_50_signal'], df['sma_signal']))
    
    return df

def analyze_signal_combination(df, combo_tuple, combo_name, ticker, price_col):
    """Analyze trading performance for a specific signal combination"""
    # Create indicator column for this combination
    if combo_name.endswith('_ONLY'):
        if 'RSI_50' in combo_name:
            df[f'ind_{combo_name}'] = df['signal_combo'].apply(lambda x: 1 if x[1] == 1 else 0)
        elif 'RSI' in combo_name and 'RSI_50' not in combo_name:
            df[f'ind_{combo_name}'] = df['signal_combo'].apply(lambda x: 1 if x[0] == 1 else 0)
        elif 'SMA' in combo_name:
            df[f'ind_{combo_name}'] = df['signal_combo'].apply(lambda x: 1 if x[2] == 1 else 0)
    elif combo_name == "RSI_RSI50_COMBO":
        df[f'ind_{combo_name}'] = df['signal_combo'].apply(lambda x: 1 if x[0] == 1 and x[1] == 1 else 0)
    elif combo_name == "RSI_SMA_COMBO":
        df[f'ind_{combo_name}'] = df['signal_combo'].apply(lambda x: 1 if x[0] == 1 and x[2] == 1 else 0)
    elif combo_name == "RSI50_SMA_COMBO":
        df[f'ind_{combo_name}'] = df['signal_combo'].apply(lambda x: 1 if x[1] == 1 and x[2] == 1 else 0)
    else:
        df[f'ind_{combo_name}'] = df['signal_combo'].apply(lambda x: 1 if x == combo_tuple else 0)

    df[f'signal_diff_{combo_name}'] = df[f'ind_{combo_name}'].diff().shift(1).fillna(0)

    # Create shifted signal diff
    df[f'signal_diff_shifted_{combo_name}'] = df[f'signal_diff_{combo_name}'].copy()
    df.loc[df[f'signal_diff_{combo_name}'] == -1, f'signal_diff_shifted_{combo_name}'] = 0
    shifted_signals = (df[f'signal_diff_{combo_name}'] == -1).shift(1).fillna(False)
    df.loc[shifted_signals, f'signal_diff_shifted_{combo_name}'] = np.where(
        df.loc[shifted_signals, f'signal_diff_{combo_name}'] == 1, 0, -1
    )
    
    df[f'signal_period_{combo_name}'] = df[f'signal_diff_shifted_{combo_name}'].ne(0).cumsum()
    
    # Get trading price 
    df[f'trad_price_{combo_name}'] = np.where(
        df[f'signal_diff_{combo_name}'].isin([1, -1]), 
        df[price_col, ticker].shift(-1), 
        np.nan
    )
    df[f'trad_price_{combo_name}'] = df[f'trad_price_{combo_name}'].fillna(method='ffill')

    # Returns - calculate both buy and sell returns
    df[f'returns_{combo_name}'] = np.where(
        df[f'signal_diff_{combo_name}'] == -1,
        df[f'trad_price_{combo_name}'].pct_change(),
        0
    )
    df[f'returns_{combo_name}'] = df[f'returns_{combo_name}'].replace([np.inf, -np.inf], 0)   
    df[f'ret_acc_{combo_name}'] = (1 + df[f'returns_{combo_name}']).cumprod()

    # Calculate sell returns (short positions)
    df[f'sell_returns_{combo_name}'] = np.where(
        df[f'signal_diff_{combo_name}'] == 1,
        -df[f'trad_price_{combo_name}'].pct_change(),  # Negative because we profit when price goes down
        0
    )
    df[f'sell_returns_{combo_name}'] = df[f'sell_returns_{combo_name}'].replace([np.inf, -np.inf], 0)
    df[f'sell_ret_acc_{combo_name}'] = (1 + df[f'sell_returns_{combo_name}']).cumprod()

    # Analyze periods
    period_returns = []
    unique_periods = df[f'signal_period_{combo_name}'].unique()
    
    for period_id in unique_periods:
        if period_id == 0:
            continue
        period_data = df[df[f'signal_period_{combo_name}'] == period_id]
        price_col_data = period_data[price_col, ticker]
        
        if len(period_data) > 1:
            price_start = price_col_data.iloc[0]
            price_end = price_col_data.iloc[-1]
            
            if period_data[f'signal_diff_shifted_{combo_name}'].iloc[0] == 1:
                # Buy signal - long position
                period_return = (price_end - price_start) / price_start
                sell_return = 0
            elif period_data[f'signal_diff_shifted_{combo_name}'].iloc[0] == -1:
                # Sell signal - short position  
                sell_return = (price_start - price_end) / price_start  # Profit when price goes down
                period_return = 0
            else:
                period_return = 0
                sell_return = 0
        else:
            period_return = 0
            sell_return = 0
            
        period_returns.append({
            'period_id': period_id,
            'start_date': period_data.index[0] if len(period_data) > 0 else None,
            'end_date': period_data.index[-1] if len(period_data) > 0 else None,
            'buy_return': period_return,
            'sell_return': sell_return
        })
    
    # Create DataFrame from period_returns
    period_df = pd.DataFrame(period_returns)
    if len(period_df) > 0:
        period_df['period_length'] = (period_df['end_date'] - period_df['start_date']).dt.days + 1
        return period_df
    else:
        return pd.DataFrame()

def create_summary_table(all_results, df):
    """Create comparative summary table with max drawdown and Sharpe ratio"""
    summary_data = []
    
    for name, result_df in all_results.items():
        if len(result_df) > 0:
            num_trades = (result_df['period_length'] > 1).sum()
            num_positive_buy = (result_df['buy_return'] > 0).sum()
            num_positive_sell = (result_df['sell_return'] > 0).sum()
            total_positive = num_positive_buy + num_positive_sell

            win_rate = total_positive / num_trades if num_trades > 0 else 0
            avg_buy_return = result_df[result_df['buy_return'] > 0]['buy_return'].mean() if num_positive_buy > 0 else 0
            avg_sell_return = result_df[result_df['sell_return'] > 0]['sell_return'].mean() if num_positive_sell > 0 else 0
            
            # Calculate total return combining both buy and sell
            total_return_buy = (1 + result_df['buy_return']).prod() - 1
            total_return_sell = (1 + result_df['sell_return']).prod() - 1
            combined_total_return = total_return_buy + total_return_sell

            # Calculate max drawdown and Sharpe ratio if strategy returns exist
            max_drawdown = 0
            sharpe_ratio = 0
            if f'returns_{name}' in df.columns:
                strategy_returns = df[f'returns_{name}'].fillna(0)
                max_drawdown = calculate_max_drawdown(strategy_returns)
                sharpe_ratio = calculate_sharpe_ratio(strategy_returns)

            # Calculate additional statistics for all periods excluding one-day trades
            multi_day_trades = result_df[result_df['period_length'] > 1]
            if len(multi_day_trades) > 0:
                longest_trade = multi_day_trades['period_length'].max()
                shortest_trade = multi_day_trades['period_length'].min()
                avg_length = multi_day_trades['period_length'].mean()
            else:
                longest_trade = shortest_trade = avg_length = 0
            
            summary_data.append({
                'Strategy': name,
                'Total Trades': num_trades,
                'Win Rate': f"{win_rate:.2%}",
                'Avg Buy Return': f"{avg_buy_return:.2%}" if num_positive_buy > 0 else "N/A",
                'Avg Sell Return': f"{avg_sell_return:.2%}" if num_positive_sell > 0 else "N/A",
                'Total Return': f"{combined_total_return:.2%}",
                'Max Drawdown': f"{max_drawdown:.2%}",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}" if sharpe_ratio != 0 else "N/A",
                'Longest Trade': longest_trade,
                'Shortest Trade': shortest_trade,
                'Avg Length': f"{avg_length:.1f}" if avg_length > 0 else "N/A"
            })
        else:
            summary_data.append({
                'Strategy': name,
                'Total Trades': 0,
                'Win Rate': "N/A",
                'Avg Buy Return': "N/A",
                'Avg Sell Return': "N/A",
                'Total Return': "N/A",
                'Max Drawdown': "N/A",
                'Sharpe Ratio': "N/A",
                'Longest Trade': "N/A",
                'Shortest Trade': "N/A",
                'Avg Length': "N/A"
            })
    
    return pd.DataFrame(summary_data)

# Run Analysis Button
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    if not selected_strategies:
        st.error("Please select at least one strategy to analyze.")
    else:
        with st.spinner("Downloading data and running analysis..."):
            # Download stock data
            df = download_data(ticker, start_date, end_date)
            
            # Download S&P 500 data for comparison
            sp500_df = download_sp500_data(start_date, end_date)
            
            if df is not None and not df.empty:
                # Calculate indicators
                df = calculate_indicators(df, ticker, price_column, rsi_period, rsi_mid_period, 
                                        sma_period, ema_period, ema_seed_period)
                
                # Define signal combinations
                signal_combinations = [
                    ((0, 0, 1), "SMA_ONLY"),
                    ((0, 1, 0), "RSI_50_ONLY"),
                    ((1, 0, 0), "RSI_ONLY"),
                    ((0, 1, 1), "SMA_RSI_50_COMBO"),
                    ((1, 0, 1), "SMA_WITH_RSI"),
                    ((1, 1, 0), "RSI_WITH_RSI_50"),
                    ((1, 1, 1), "RSI_RSI50_SMA_COMBO"),
                    ("CUSTOM", "RSI_RSI50_COMBO"),
                    ("CUSTOM", "RSI_SMA_COMBO"),
                    ("CUSTOM", "RSI50_SMA_COMBO"),
                ]
                
                # Run analysis for selected strategies
                all_results = {}
                
                for combo, name in signal_combinations:
                    if name in selected_strategies:
                        try:
                            result_df = analyze_signal_combination(df, combo, name, ticker, price_column)
                            all_results[name] = result_df
                        except Exception as e:
                            st.error(f"Error analyzing {name}: {e}")
                
                # Store results in session state
                st.session_state['results'] = all_results
                st.session_state['df'] = df
                st.session_state['sp500_df'] = sp500_df
                st.session_state['ticker'] = ticker
                st.session_state['price_column'] = price_column

# Display results if available
if 'results' in st.session_state:
    all_results = st.session_state['results']
    df = st.session_state['df']
    sp500_df = st.session_state.get('sp500_df', None)
    ticker = st.session_state['ticker']
    price_column = st.session_state['price_column']
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "📈 Charts", "🔍 Detailed Analysis", "📋 Raw Data", "📉 Risk Analysis"])
    
    with tab1:
        st.header("📊 Strategy Performance Summary")
        
        if all_results:
            summary_df = create_summary_table(all_results, df)
            
            # Display summary table
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Performance metrics cards
            st.subheader("🏆 Top Performers")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Filter out N/A values for ranking
            valid_returns = summary_df[summary_df['Total Return'] != 'N/A'].copy()
            if not valid_returns.empty:
                valid_returns['Total Return Numeric'] = valid_returns['Total Return'].str.rstrip('%').astype(float)
                best_strategy = valid_returns.loc[valid_returns['Total Return Numeric'].idxmax()]
                
                with col1:
                    st.metric(
                        "🥇 Best Total Return",
                        best_strategy['Strategy'],
                        best_strategy['Total Return']
                    )
            
            valid_trades = summary_df[summary_df['Total Trades'] != 0]
            if not valid_trades.empty:
                most_active = valid_trades.loc[valid_trades['Total Trades'].idxmax()]
                
                with col2:
                    st.metric(
                        "📈 Most Active Strategy",
                        most_active['Strategy'],
                        f"{most_active['Total Trades']} trades"
                    )
            
            valid_sharpe = summary_df[summary_df['Sharpe Ratio'] != 'N/A'].copy()
            if not valid_sharpe.empty:
                valid_sharpe['Sharpe Ratio Numeric'] = valid_sharpe['Sharpe Ratio'].astype(float)
                best_sharpe = valid_sharpe.loc[valid_sharpe['Sharpe Ratio Numeric'].idxmax()]
                
                with col3:
                    st.metric(
                        "⚡ Best Sharpe Ratio",
                        best_sharpe['Strategy'],
                        best_sharpe['Sharpe Ratio']
                    )
            
            valid_drawdown = summary_df[summary_df['Max Drawdown'] != 'N/A'].copy()
            if not valid_drawdown.empty:
                valid_drawdown['Max Drawdown Numeric'] = valid_drawdown['Max Drawdown'].str.rstrip('%').astype(float)
                best_drawdown = valid_drawdown.loc[valid_drawdown['Max Drawdown Numeric'].idxmax()]  # Closest to 0
                
                with col4:
                    st.metric(
                        "🛡️ Lowest Drawdown",
                        best_drawdown['Strategy'],
                        best_drawdown['Max Drawdown']
                    )
    
    with tab2:
        st.header("📈 Performance Charts")
        
        # Cumulative returns chart
        fig = go.Figure()
        
        # Add market returns
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['returns_acc'],
            name=f'{ticker} Buy & Hold',
            line=dict(color='red', width=2)
        ))
        
        # Add strategy returns
        colors = px.colors.qualitative.Set1
        for i, (name, result_df) in enumerate(all_results.items()):
            if f'ret_acc_{name}' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[f'ret_acc_{name}'],
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
        
        fig.update_layout(
            title=f"Cumulative Returns Comparison - {ticker}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual strategy performance
        st.subheader("Individual Strategy Analysis")
        
        selected_strategy = st.selectbox(
            "Select Strategy for Detailed Chart",
            list(all_results.keys())
        )
        
        if selected_strategy and f'ret_acc_{selected_strategy}' in df.columns:
            # Create subplot with price and strategy performance
            fig_detail = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f'{ticker} Price', f'{selected_strategy} Performance'],
                vertical_spacing=0.1
            )
            
            # Price chart
            fig_detail.add_trace(
                go.Scatter(x=df.index, y=df[price_column, ticker], name=f'{ticker} Price'),
                row=1, col=1
            )
            
            # Strategy performance
            fig_detail.add_trace(
                go.Scatter(x=df.index, y=df[f'ret_acc_{selected_strategy}'], 
                          name=f'{selected_strategy}', line=dict(color='green')),
                row=2, col=1
            )
            
            fig_detail.add_trace(
                go.Scatter(x=df.index, y=df['returns_acc'], 
                          name='Buy & Hold', line=dict(color='red')),
                row=2, col=1
            )
            
            fig_detail.update_layout(height=700, showlegend=True)
            st.plotly_chart(fig_detail, use_container_width=True)
    
    with tab3:
        st.header("🔍 Detailed Strategy Analysis")
        
        strategy_detail = st.selectbox(
            "Select Strategy for Detailed Analysis",
            list(all_results.keys()),
            key="detail_strategy"
        )
        
        if strategy_detail and strategy_detail in all_results:
            result_df = all_results[strategy_detail]
            
            if not result_df.empty:
                st.subheader(f"Trade Details - {strategy_detail}")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    min_length = st.number_input("Minimum Trade Length (days)", min_value=1, value=1)
                with col2:
                    trade_type = st.selectbox("Trade Type", ["All", "Buy Trades", "Sell Trades"])
                
                # Filter data
                filtered_df = result_df[result_df['period_length'] >= min_length].copy()
                
                if trade_type == "Buy Trades":
                    filtered_df = filtered_df[filtered_df['buy_return'] != 0]
                elif trade_type == "Sell Trades":
                    filtered_df = filtered_df[filtered_df['sell_return'] != 0]
                
                # Display filtered trades
                if not filtered_df.empty:
                    st.dataframe(
                        filtered_df[['period_id', 'start_date', 'end_date', 'period_length', 
                                   'buy_return', 'sell_return']].round(6),
                        use_container_width=True
                    )
                    
                    # Trade statistics
                    st.subheader("Trade Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Trades", len(filtered_df))
                    with col2:
                        avg_length = filtered_df['period_length'].mean()
                        st.metric("Avg Trade Length", f"{avg_length:.1f} days")
                    with col3:
                        buy_trades = filtered_df[filtered_df['buy_return'] > 0]
                        win_rate = len(buy_trades) / len(filtered_df) if len(filtered_df) > 0 else 0
                        st.metric("Win Rate", f"{win_rate:.1%}")
                    with col4:
                        total_return = (1 + filtered_df['buy_return']).prod() - 1
                        st.metric("Total Return", f"{total_return:.2%}")
                else:
                    st.info("No trades match the selected criteria.")
            else:
                st.info(f"No data available for {strategy_detail}")
    
    with tab4:
        st.header("📋 Raw Data")
        
        # Show indicator values
        st.subheader("Technical Indicators")
        
        # Debug: Show actual DataFrame columns
        st.write("**Debug - DataFrame columns:**")
        st.write(f"Total columns: {len(df.columns)}")
        st.write("Sample columns:", list(df.columns[:10]))
        
        # Build display columns list more carefully
        final_columns = []
        
        # Add price column (MultiIndex) - check first
        price_col_tuple = (price_column, ticker)
        if price_col_tuple in df.columns:
            final_columns.append(price_col_tuple)
            st.write(f"✅ Added price column: {price_col_tuple}")
        else:
            st.write(f"❌ Price column not found: {price_col_tuple}")
        
        # Add technical indicator columns (single level) - check each one
        tech_indicators = ['rsi', 'rsi_mid', 'sma', 'ema_manual', 'rsi_signal', 'rsi_50_signal', 'sma_signal', 'returns']
        st.write("**Technical Indicators Check:**")
        for col in tech_indicators:
            if col in df.columns:
                final_columns.append(col)
                st.write(f"✅ {col}")
            else:
                st.write(f"❌ {col}")
        
        # Add strategy-specific columns - check each one
        if 'results' in st.session_state and st.session_state['results']:
            st.write("**Strategy Columns Check:**")
            for strategy in st.session_state['results'].keys():
                strategy_cols = [f'returns_{strategy}', f'ret_acc_{strategy}', f'ind_{strategy}']
                for col in strategy_cols:
                    if col in df.columns:
                        final_columns.append(col)
                        st.write(f"✅ {col}")
                    else:
                        st.write(f"❌ {col}")
        
        # Final safety check
        st.write(f"**Final columns to display ({len(final_columns)}):**")
        st.write(final_columns)
        
        if final_columns:
            try:
                # Create a copy of the dataframe with selected columns
                display_df = df[final_columns].copy()
                
                # Flatten column names for better display
                if hasattr(display_df.columns, 'levels'):
                    # Handle MultiIndex columns
                    new_columns = []
                    for col in display_df.columns:
                        if isinstance(col, tuple):
                            new_columns.append(f"{col[0]}_{col[1]}")
                        else:
                            new_columns.append(str(col))
                    display_df.columns = new_columns
                
                st.dataframe(
                    display_df.tail(100).round(6),
                    use_container_width=True
                )
                
                # Download button for full data
                csv = display_df.to_csv()
                st.download_button(
                    label="📥 Download Full Dataset as CSV",
                    data=csv,
                    file_name=f"{ticker}_trading_analysis.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error displaying data: {e}")
                st.write("**All DataFrame columns:**")
                st.write(list(df.columns))
        else:
            st.warning("No data columns available to display.")
            st.write("**All available DataFrame columns:**")
            st.write(list(df.columns))
    
    with tab5:
        st.header("📉 Risk Analysis")
        
        # Calculate benchmark metrics
        if sp500_df is not None and not sp500_df.empty:
            try:
                # Debug: Show available columns
                st.write("**S&P 500 Debug - Available columns:**")
                available_cols = list(sp500_df.columns)
                st.write(f"Columns: {available_cols}")
                st.write(f"Number of columns: {len(available_cols)}")
                
                # S&P 500 metrics - use safe column access
                sp500_price_data = safe_column_access(sp500_df, ['Adj Close', 'Close', ('Adj Close', '^GSPC'), ('Close', '^GSPC')])
                
                if sp500_price_data is None:
                    st.write("**Trying manual column access...**")
                    # Try manual access for debugging
                    if available_cols:
                        first_col = available_cols[0]
                        st.write(f"Trying first available column: {first_col}")
                        sp500_price_data = sp500_df[first_col]
                    else:
                        raise ValueError("No columns found in S&P 500 data")
                
                if sp500_price_data is None:
                    raise ValueError("Unable to find price column in S&P 500 data")
                
                st.write(f"**Successfully accessed S&P 500 data:** {type(sp500_price_data)}")
                
                sp500_returns = sp500_price_data.pct_change().dropna()
                
                # Validate sp500_returns before proceeding
                if sp500_returns.empty or len(sp500_returns) == 0:
                    raise ValueError("Unable to extract valid S&P 500 returns data")
                
                sp500_max_drawdown = calculate_max_drawdown(sp500_returns)
                sp500_sharpe = calculate_sharpe_ratio(sp500_returns)
                sp500_total_return = (1 + sp500_returns).prod() - 1
                
                # Stock metrics - use safe column access for consistency
                stock_price_data = safe_column_access(df, [(price_column, ticker)])
                
                if stock_price_data is None:
                    raise ValueError(f"Unable to find {price_column} column for {ticker}")
                
                stock_returns = stock_price_data.pct_change().dropna()
                
                # Validate stock returns
                if stock_returns.empty or len(stock_returns) == 0:
                    raise ValueError("Unable to extract valid stock returns data")
                
                stock_max_drawdown = calculate_max_drawdown(stock_returns)
                stock_sharpe = calculate_sharpe_ratio(stock_returns)
                stock_total_return = (1 + stock_returns).prod() - 1
                
                st.subheader("📊 Benchmark Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("S&P 500 Total Return", f"{sp500_total_return:.2%}")
                    st.metric("S&P 500 Max Drawdown", f"{sp500_max_drawdown:.2%}")
                    st.metric("S&P 500 Sharpe Ratio", f"{sp500_sharpe:.2f}")
                
                with col2:
                    st.metric(f"{ticker} Total Return", f"{stock_total_return:.2%}")
                    st.metric(f"{ticker} Max Drawdown", f"{stock_max_drawdown:.2%}")
                    st.metric(f"{ticker} Sharpe Ratio", f"{stock_sharpe:.2f}")
                
                with col3:
                    return_diff = stock_total_return - sp500_total_return
                    drawdown_diff = stock_max_drawdown - sp500_max_drawdown
                    sharpe_diff = stock_sharpe - sp500_sharpe
                    
                    st.metric("Return vs S&P 500", f"{return_diff:.2%}")
                    st.metric("Drawdown vs S&P 500", f"{drawdown_diff:.2%}")
                    st.metric("Sharpe vs S&P 500", f"{sharpe_diff:.2f}")
            except Exception as e:
                st.error(f"Error calculating benchmark metrics: {e}")
                st.info("Unable to load S&P 500 data for comparison")
                # Enhanced debug information
                if sp500_df is not None:
                    st.write("**S&P 500 Debug Information:**")
                    st.write(f"DataFrame shape: {sp500_df.shape}")
                    st.write(f"Column structure: {type(sp500_df.columns)}")
                    st.write(f"Columns: {list(sp500_df.columns)}")
                    if hasattr(sp500_df.columns, 'levels'):
                        st.write(f"MultiIndex levels: {sp500_df.columns.levels}")
                    
                    # Show first few rows for debugging
                    st.write("First 3 rows:")
                    st.write(sp500_df.head(3))
                else:
                    st.write("sp500_df is None")
        else:
            st.info("S&P 500 data not available for comparison")
        
        # Strategy risk comparison
        st.subheader("🎯 Strategy Risk Metrics")
        
        risk_data = []
        for name in all_results.keys():
            if f'returns_{name}' in df.columns:
                try:
                    strategy_returns = df[f'returns_{name}'].fillna(0)
                    max_dd = calculate_max_drawdown(strategy_returns)
                    sharpe = calculate_sharpe_ratio(strategy_returns)
                    total_ret = (1 + strategy_returns).prod() - 1
                    volatility = strategy_returns.std() * np.sqrt(252)
                    
                    risk_data.append({
                        'Strategy': name,
                        'Total Return': f"{total_ret:.2%}",
                        'Volatility': f"{volatility:.2%}",
                        'Max Drawdown': f"{max_dd:.2%}",
                        'Sharpe Ratio': f"{sharpe:.2f}",
                        'Return/Risk': f"{total_ret/max(volatility, 0.001):.2f}"
                    })
                except Exception as e:
                    st.warning(f"Error calculating metrics for {name}: {e}")
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        # Risk-Return scatter plot
        if risk_data:
            st.subheader("📈 Risk-Return Scatter Plot")
            
            try:
                fig_risk = go.Figure()
                
                for item in risk_data:
                    ret = float(item['Total Return'].rstrip('%')) / 100
                    vol = float(item['Volatility'].rstrip('%')) / 100
                    
                    fig_risk.add_trace(go.Scatter(
                        x=[vol],
                        y=[ret],
                        mode='markers+text',
                        name=item['Strategy'],
                        text=[item['Strategy']],
                        textposition="top center",
                        marker=dict(size=10)
                    ))
                
                fig_risk.update_layout(
                    title="Risk vs Return Analysis",
                    xaxis_title="Volatility (Risk)",
                    yaxis_title="Total Return",
                    showlegend=False,
                    height=500
                )
                
                st.plotly_chart(fig_risk, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating risk-return plot: {e}")

else:
    st.info("👆 Configure your analysis parameters in the sidebar and click 'Run Analysis' to get started!")
    
    # Show sample configuration
    st.subheader("🎯 Quick Start Guide")
    st.markdown("""
    1. **Configure Parameters**: Use the sidebar to set your ticker symbol, date range, and indicator parameters
    2. **Select Strategies**: Choose which trading strategies you want to backtest
    3. **Run Analysis**: Click the 'Run Analysis' button to process the data
    4. **Review Results**: Explore the results in the different tabs:
       - **Summary**: Overview of strategy performance
       - **Charts**: Visual comparison of strategies
       - **Detailed Analysis**: Deep dive into individual trades
       - **Raw Data**: Access to underlying data
    
    **Popular Strategy Combinations:**
    - `RSI_ONLY`: Pure RSI-based signals
    - `RSI_RSI50_COMBO`: RSI combined with RSI mid-point
    - `SMA_ONLY`: Simple moving average crossover
    - `RSI_RSI50_SMA_COMBO`: All three indicators combined
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and Plotly for interactive trading strategy analysis*")
