import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os
import pandas as pd

# --- Set up the logging configuration ---
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def calculate_ma(data, window):
    """Calculates a Simple Moving Average (SMA) from a list of data dictionaries."""
    if len(data) < window:
        return None
    closes = [d['Close'] for d in data[-window:]]
    return sum(closes) / window

def calculate_rsi(data, period):
    """Calculates the Relative Strength Index (RSI) from a list of data dictionaries."""
    if len(data) < period + 1:
        return None
    
    closes = np.array([d['Close'] for d in data])
    diff = np.diff(closes[-period-1:])
    
    gains = diff[diff > 0]
    losses = -diff[diff < 0]

    avg_gain = np.mean(gains) if gains.size > 0 else 0
    avg_loss = np.mean(losses) if losses.size > 0 else 0
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window, std_dev):
    """Calculates Bollinger Bands from a list of data dictionaries."""
    if len(data) < window:
        return None, None
    
    closes = np.array([d['Close'] for d in data[-window:]])
    rolling_mean = np.mean(closes)
    rolling_std = np.std(closes)
    
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

def backtest_strategy(historical_data, short_ma, long_ma, rsi_period, bb_std_dev):
    """
    Backtests the strategy using a for-loop and basic Python data types.
    """
    trades = []
    positions = []
    in_position = False
    entry_price = 0.0
    
    for i in range(len(historical_data)):
        current_data = historical_data[:i+1]
        
        # Calculate indicators for the current day
        current_close = current_data[-1]['Close']
        current_short_ma = calculate_ma(current_data, short_ma)
        current_long_ma = calculate_ma(current_data, long_ma)
        current_rsi = calculate_rsi(current_data, rsi_period)
        current_upper_bb, current_lower_bb = calculate_bollinger_bands(current_data, 20, bb_std_dev)
        
        # Ensure we have enough data for a signal
        if current_short_ma is None or current_long_ma is None or current_rsi is None or current_upper_bb is None:
            positions.append(0)
            continue
            
        # Get previous day's data for comparison
        prev_data = historical_data[:i]
        prev_short_ma = calculate_ma(prev_data, short_ma)
        prev_long_ma = calculate_ma(prev_data, long_ma)
        
        # Guard against None values if no previous data exists
        if prev_short_ma is None or prev_long_ma is None:
            positions.append(0)
            continue

        # Generate trading signals
        ma_cross_buy = (current_short_ma > current_long_ma) and (prev_short_ma <= prev_long_ma)
        rsi_buy_conf = current_rsi > 50
        bb_buy_conf = current_close < current_lower_bb

        buy_signal = ma_cross_buy and rsi_buy_conf and bb_buy_conf
        
        ma_cross_sell = (current_short_ma < current_long_ma) and (prev_short_ma >= prev_long_ma)
        bb_sell_conf = current_close > current_upper_bb
        
        sell_signal = ma_cross_sell or bb_sell_conf
        
        # Execute trades
        if not in_position and buy_signal:
            entry_price = current_close
            in_position = True
            trades.append({'Type': 'Buy', 'Date': historical_data[i]['Date'], 'Price': entry_price})
            
        elif in_position and sell_signal:
            sell_price = current_close
            profit_pct = ((sell_price - entry_price) / entry_price) * 100
            in_position = False
            trades.append({'Type': 'Sell', 'Date': historical_data[i]['Date'], 'Price': sell_price, 'Profit_Pct': profit_pct})
        
        positions.append(1 if in_position else 0)

    return trades, positions

def advanced_strategy_with_optimization(ticker, start_date, end_date):
    """
    Performs an advanced backtest with parameter optimization and detailed metrics.
    """
    print("Script started and logging configured. Proceeding with data download...")
    logging.info(f"Starting advanced optimization for {ticker} from {start_date} to {end_date}...")
    logging.info(f"Detailed logs will be written to '{LOG_FILE}'")

    try:
        data_df = yf.download(ticker, start=start_date, end=end_date)
        if data_df.empty:
            logging.error(f"Error: No data found for ticker {ticker}. Please check the symbol and date range.")
            return
            
        # Convert DataFrame to a list of dictionaries for non-pandas logic
        historical_data = []
        for index, row in data_df.iterrows():
            historical_data.append({
                'Date': index.strftime('%Y-%m-%d'),
                'Close': row['Close']
            })

    except Exception as e:
        logging.error(f"An error occurred while downloading data: {e}")
        return

    results = []
    
    short_ma_range = [20, 50]
    long_ma_range = [100, 200]
    rsi_period_range = [10, 14, 21]
    bb_std_dev_range = [1, 2]

    best_sharpe_ratio = -np.inf
    best_strategy_params = None

    for short_ma in short_ma_range:
        for long_ma in long_ma_range:
            for rsi_period in rsi_period_range:
                for bb_std_dev in bb_std_dev_range:
                    if short_ma >= long_ma:
                        continue
                    
                    logging.info(f"\n--- Testing Strategy: MA({short_ma},{long_ma}), RSI({rsi_period}), BB_Std({bb_std_dev}) ---")
                    
                    trades, positions = backtest_strategy(historical_data, short_ma, long_ma, rsi_period, bb_std_dev)
                    
                    if not trades:
                        continue

                    # Calculate performance metrics
                    profit_pcts = [t['Profit_Pct'] for t in trades if 'Profit_Pct' in t]
                    total_trades = len(profit_pcts)
                    if total_trades == 0:
                        continue
                        
                    win_rate = sum(1 for p in profit_pcts if p > 0) / total_trades * 100
                    total_return = sum(profit_pcts)
                    
                    avg_return = np.mean(profit_pcts) if profit_pcts else 0
                    std_return = np.std(profit_pcts) if profit_pcts else 0
                    sharpe_ratio = avg_return / std_return if std_return > 0 else -1

                    new_result = {
                        'Short_MA': short_ma,
                        'Long_MA': long_ma,
                        'RSI_Period': rsi_period,
                        'Bollinger_StdDev': bb_std_dev,
                        'Total_Return_Pct': total_return,
                        'Sharpe_Ratio': sharpe_ratio,
                        'Win_Rate_Pct': win_rate,
                        'Total_Trades': total_trades
                    }
                    results.append(new_result)

                    if sharpe_ratio > best_sharpe_ratio:
                        best_sharpe_ratio = sharpe_ratio
                        best_strategy_params = new_result

    if not results or best_strategy_params is None:
        logging.info("\n--- No profitable strategies found. ---")
        return
        
    logging.info("\n--- Optimization Complete ---")
    logging.info("Best Performing Strategy based on Sharpe Ratio:")
    logging.info(str(best_strategy_params))
    
    # Visualization using pandas for efficiency
    logging.info("\nGenerating chart for the best performing strategy...")
    best_short_ma = best_strategy_params['Short_MA']
    best_long_ma = best_strategy_params['Long_MA']
    best_rsi_period = best_strategy_params['RSI_Period']
    best_bb_std_dev = best_strategy_params['Bollinger_StdDev']

    final_data_df = yf.download(ticker, start=start_date, end=end_date)
    final_data_df['Short_MA'] = final_data_df['Close'].rolling(window=best_short_ma).mean()
    final_data_df['Long_MA'] = final_data_df['Close'].rolling(window=best_long_ma).mean()
    final_data_df['Rolling_Mean'] = final_data_df['Close'].rolling(window=20).mean()
    final_data_df['Rolling_Std'] = final_data_df['Close'].rolling(window=20).std()
    final_data_df['Upper_BB'] = final_data_df['Rolling_Mean'] + (final_data_df['Rolling_Std'] * best_bb_std_dev)
    final_data_df['Lower_BB'] = final_data_df['Rolling_Mean'] - (final_data_df['Rolling_Std'] * best_bb_std_dev)

    delta = final_data_df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=best_rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=best_rsi_period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    final_data_df['RSI'] = 100 - (100 / (1 + rs))

    # Re-apply the backtest logic to the pandas DataFrame for plotting
    final_data_df['Signal'] = 0
    in_position = False
    
    for i in range(1, len(final_data_df)):
        if pd.isna(final_data_df.iloc[i]['Short_MA']) or pd.isna(final_data_df.iloc[i]['Long_MA']):
            continue

        ma_cross_buy = (final_data_df.iloc[i]['Short_MA'] > final_data_df.iloc[i]['Long_MA']) and (final_data_df.iloc[i-1]['Short_MA'] <= final_data_df.iloc[i-1]['Long_MA'])
        rsi_buy_conf = final_data_df.iloc[i]['RSI'] > 50
        bb_buy_conf = final_data_df.iloc[i]['Close'] < final_data_df.iloc[i]['Lower_BB']
        buy_signal = ma_cross_buy and rsi_buy_conf and bb_buy_conf

        ma_cross_sell = (final_data_df.iloc[i]['Short_MA'] < final_data_df.iloc[i]['Long_MA']) and (final_data_df.iloc[i-1]['Short_MA'] >= final_data_df.iloc[i-1]['Long_MA'])
        bb_sell_conf = final_data_df.iloc[i]['Close'] > final_data_df.iloc[i]['Upper_BB']
        sell_signal = ma_cross_sell or bb_sell_conf
        
        if not in_position and buy_signal:
            final_data_df.loc[final_data_df.index[i], 'Signal'] = 1
            in_position = True
        elif in_position and sell_signal:
            final_data_df.loc[final_data_df.index[i], 'Signal'] = -1
            in_position = False
    
    final_data_df.dropna(inplace=True)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(final_data_df['Close'], label='Close Price', alpha=0.5)
    ax1.plot(final_data_df['Short_MA'], label=f'Short MA ({best_short_ma} days)', color='gold')
    ax1.plot(final_data_df['Long_MA'], label=f'Long MA ({best_long_ma} days)', color='dodgerblue')
    ax1.plot(final_data_df['Upper_BB'], label=f'Upper BB ({best_bb_std_dev} StdDev)', linestyle='--', color='purple')
    ax1.plot(final_data_df['Lower_BB'], label=f'Lower BB ({best_bb_std_dev} StdDev)', linestyle='--', color='purple')
    ax1.plot(final_data_df.loc[final_data_df['Signal'] == 1].index, final_data_df['Close'][final_data_df['Signal'] == 1], '^', markersize=12, color='lime', label='Buy Signal')
    ax1.plot(final_data_df.loc[final_data_df['Signal'] == -1].index, final_data_df['Close'][final_data_df['Signal'] == -1], 'v', markersize=12, color='red', label='Sell Signal')
    ax1.set_title(f'{ticker} Best Strategy Performance', fontsize=16)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    

    ax2.plot(final_data_df['RSI'], label='RSI', color='cyan')
    ax2.axhline(y=50, color='gray', linestyle='--')
    ax2.fill_between(final_data_df.index, y1=30, y2=70, color='gray', alpha=0.2, label='Normal Range')
    ax2.set_title('Relative Strength Index (RSI)', fontsize=14)
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    advanced_strategy_with_optimization(
        ticker='AAPL',
        start_date='2018-01-01',
        end_date='2023-01-01'
    )
