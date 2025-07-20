# 📈 Trading Strategy Backtesting Dashboard

This project provides comprehensive tools for backtesting various trading strategies using technical indicators. You can analyze different combinations of RSI, SMA, and EMA signals to evaluate their performance.

## 🚀 Quick Start

### Option 1: Streamlit Web Dashboard (Recommended)
```bash
# Install requirements
pip install -r requirements.txt

# Run the dashboard
streamlit run trading_dashboard.py

# Or use the batch file (Windows)
run_dashboard.bat
```

Open your browser to `http://localhost:8501`

### Option 2: Jupyter Notebook
Open `Multi trade.ipynb` and run all cells. The notebook includes:
- Complete backtesting analysis
- Interactive widgets for strategy comparison
- Detailed period analysis

## 📊 Dashboard Features

### Configuration Options
- **Ticker Symbol**: Choose any stock (MSFT, AAPL, GOOGL, etc.)
- **Date Range**: Customize analysis period
- **Price Column**: Select from Adj Close, Close, Open, High, Low
- **Indicator Parameters**:
  - RSI period and RSI mid period
  - SMA and EMA periods and seeds

### Available Strategies
1. **RSI_ONLY**: Pure RSI-based signals
2. **RSI_50_ONLY**: RSI mid-point signals
3. **SMA_ONLY**: Moving average crossover
4. **RSI_WITH_RSI_50**: Combined RSI signals
5. **SMA_WITH_RSI**: SMA + RSI combination
6. **SMA_RSI_50_COMBO**: SMA + RSI mid combination
7. **RSI_RSI50_SMA_COMBO**: All three indicators
8. **RSI_RSI50_COMBO**: RSI + RSI mid custom combo
9. **RSI_SMA_COMBO**: RSI + SMA custom combo
10. **RSI50_SMA_COMBO**: RSI mid + SMA custom combo

### Dashboard Tabs

#### 📊 Summary Tab
- Comparative performance table showing:
  - Total number of trades
  - Win rate percentage
  - Average return per trade
  - Total cumulative return
  - Trade length statistics
- Top performer metrics cards

#### 📈 Charts Tab
- Interactive cumulative returns comparison
- Individual strategy detailed charts
- Price vs strategy performance subplots

#### 🔍 Detailed Analysis Tab
- Trade-by-trade breakdown
- Filtering options:
  - Minimum trade length
  - Trade type (Buy/Sell/All)
- Trade statistics and metrics

#### 📋 Raw Data Tab
- Technical indicator values
- Signal combinations
- Strategy-specific returns
- CSV download capability

## 🔧 Technical Indicators

### RSI (Relative Strength Index)
- Configurable period (default: 30)
- RSI mid-point calculation (rolling average)
- Signals based on RSI > RSI_mid

### SMA/EMA (Moving Averages)
- Simple Moving Average (configurable period)
- Exponential Moving Average with seed period
- Signals based on EMA > SMA crossover

### Signal Combinations
The system generates binary signals from each indicator and analyzes all possible combinations:
- Signal tuple format: (rsi_signal, rsi_50_signal, sma_signal)
- Each element is 0 (inactive) or 1 (active)

## 📈 Strategy Logic

### Buy Signals
- Generated when signal combination becomes active (0 → 1)
- Position held until sell signal

### Sell Signals  
- Generated when signal combination becomes inactive (1 → 0)
- Returns calculated from entry to exit price

### Performance Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Return**: Mean return of winning trades
- **Total Return**: Cumulative strategy performance
- **Trade Statistics**: Length, frequency, and timing analysis

## 🛠️ Installation

### Requirements
```bash
streamlit==1.39.0
plotly==5.24.1
yfinance==0.2.44
pandas-ta==0.3.14b0
pandas>=1.5.0
numpy>=1.21.0
```

### Setup
```bash
# Clone or download the project files
# Install dependencies
pip install -r requirements.txt

# For Jupyter notebook widgets (optional)
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## 📁 Project Structure

```
Saudi/
├── Multi trade.ipynb          # Main analysis notebook
├── trading_dashboard.py       # Streamlit web dashboard
├── requirements.txt           # Python dependencies
├── run_dashboard.bat         # Windows launcher script
└── README.md                 # This file
```

## 🎯 Usage Examples

### Streamlit Dashboard
1. Launch the dashboard
2. Enter ticker symbol (e.g., "AAPL")
3. Adjust date range and parameters
4. Select strategies to compare
5. Click "Run Analysis"
6. Explore results in different tabs

### Notebook Analysis
1. Open `Multi trade.ipynb`
2. Run all cells to perform full analysis
3. Use interactive widgets for custom analysis
4. Modify parameters and re-run specific sections

## 📊 Sample Analysis Output

The dashboard generates comparative tables like:

```
Strategy             | Total Trades | Win Rate | Avg Return | Total Return | Avg Length
RSI_ONLY            | 45          | 64.44%   | 8.23%      | 156.78%     | 23.4
RSI_50_ONLY         | 38          | 71.05%   | 7.89%      | 142.56%     | 28.7
SMA_ONLY            | 52          | 59.62%   | 6.45%      | 134.23%     | 19.8
RSI_RSI50_COMBO     | 23          | 78.26%   | 9.87%      | 198.45%     | 34.2
```

## 🤝 Contributing

Feel free to enhance the dashboard by:
- Adding new technical indicators
- Implementing additional strategy combinations
- Improving visualization options
- Adding risk metrics and analysis

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough research and consider consulting financial professionals before making investment decisions.

## 📞 Support

For questions or issues:
1. Check the notebook output for error messages
2. Verify all required packages are installed
3. Ensure data connectivity for Yahoo Finance API
4. Review parameter ranges and strategy selections
