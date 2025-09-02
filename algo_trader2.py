import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

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

def apply_vectorized_strategy(data, short_ma, long_ma, rsi_period, bb_std_dev):
    """
    Applies the trading strategy using vectorized operations.
    Returns data with signals, positions, and trades.
    """
    strategy_data = data.copy()

    # 1. Calculate All Indicators (Vectorized)
    strategy_data['Short_MA'] = strategy_data['Close'].rolling(window=short_ma).mean()
    strategy_data['Long_MA'] = strategy_data['Close'].rolling(window=long_ma).mean()
    
    delta = strategy_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    strategy_data['RSI'] = 100 - (100 / (1 + rs))

    strategy_data['Rolling_Mean'] = strategy_data['Close'].rolling(window=20).mean()
    strategy_data['Rolling_Std'] = strategy_data['Close'].rolling(window=20).std()
    strategy_data['Upper_BB'] = strategy_data['Rolling_Mean'] + (strategy_data['Rolling_Std'] * bb_std_dev)
    strategy_data['Lower_BB'] = strategy_data['Rolling_Mean'] - (strategy_data['Rolling_Std'] * bb_std_dev)

    # 2. Define Buy/Sell Conditions (Vectorized)
    # The fix is to perform all comparisons *before* dropping any rows.
    # The .shift(1) is applied here to avoid using the same-day data for signals.
    buy_conditions = (strategy_data['Short_MA'] > strategy_data['Long_MA']) & \
                     (strategy_data['Short_MA'].shift(1) <= strategy_data['Long_MA'].shift(1)) & \
                     (strategy_data['RSI'] > 50) & \
                     (strategy_data['Close'] < strategy_data['Lower_BB'])

    sell_conditions = ((strategy_data['Short_MA'] < strategy_data['Long_MA']) & \
                      (strategy_data['Short_MA'].shift(1) >= strategy_data['Long_MA'].shift(1))) | \
                      (strategy_data['Close'] > strategy_data['Upper_BB'])

    # 3. Drop NaN values *after* defining all conditions.
    strategy_data.dropna(inplace=True)

    # 4. Generate Trading Signals (Vectorized)
    strategy_data['Signal'] = 0
    strategy_data.loc[buy_conditions, 'Signal'] = 1
    strategy_data.loc[sell_conditions, 'Signal'] = -1

    # 5. Generate Positions (Vectorized)
    strategy_data['Position'] = strategy_data['Signal'].replace(to_replace=0, method='ffill')
    strategy_data['Position'] = strategy_data['Position'].shift(1)
    
    # Generate a list of trades for detailed analysis
    trades = []
    in_position = False
    entry_price = 0.0
    
    for i in range(1, len(strategy_data)):
        current_signal = strategy_data['Signal'].iloc[i]
        current_close = strategy_data['Close'].iloc[i]
        trade_date = strategy_data.index[i].strftime('%Y-%m-%d')
        
        if current_signal == 1 and not in_position:
            entry_price = current_close
            in_position = True
            trades.append({'Type': 'Buy', 'Date': trade_date, 'Price': entry_price})
            logging.info(f"    - BUY Signal on {trade_date} at price ${entry_price:.2f}")

        elif current_signal == -1 and in_position:
            sell_price = current_close
            profit_pct = ((sell_price - entry_price) / entry_price) * 100
            in_position = False
            reason = "MA Cross" if (strategy_data['Short_MA'].iloc[i] < strategy_data['Long_MA'].iloc[i]) else "Above BB"
            trades.append({'Type': 'Sell', 'Date': trade_date, 'Price': sell_price, 'Profit_Pct': profit_pct})
            logging.info(f"    - SELL Signal on {trade_date} at price ${sell_price:.2f}. P/L: {profit_pct:.2f}% ({reason})")
    
    return strategy_data, trades

def advanced_strategy_with_optimization(ticker, start_date, end_date):
    # This function remains mostly the same, now calling the vectorized function.
    print("Script started and logging configured. Proceeding with data download...")
    logging.info(f"Starting advanced optimization for {ticker} from {start_date} to {end_date}...")
    logging.info(f"Detailed logs will be written to '{LOG_FILE}'")

    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            logging.error(f"Error: No data found for ticker {ticker}. Please check the symbol and date range.")
            return
    except Exception as e:
        logging.error(f"An error occurred while downloading data: {e}")
        return

    results = pd.DataFrame(columns=['Short_MA', 'Long_MA', 'RSI_Period', 'Bollinger_StdDev', 
                                    'Total_Return_Pct', 'Sharpe_Ratio', 'Max_Drawdown_Pct', 
                                    'Win_Rate_Pct', 'Total_Trades'])
    
    short_ma_range = [20, 50]
    long_ma_range = [100, 200]
    rsi_period_range = [10, 14, 21]
    bb_std_dev_range = [1, 2]

    for short_ma in short_ma_range:
        for long_ma in long_ma_range:
            for rsi_period in rsi_period_range:
                for bb_std_dev in bb_std_dev_range:
                    if short_ma >= long_ma:
                        continue

                    logging.info(f"\n--- Testing Strategy: MA({short_ma},{long_ma}), RSI({rsi_period}), BB_Std({bb_std_dev}) ---")
                    
                    strategy_data, trades = apply_vectorized_strategy(data, short_ma, long_ma, rsi_period, bb_std_dev)

                    # 4. Calculate Performance Metrics
                    strategy_data['Strategy_Returns'] = strategy_data['Close'].pct_change() * strategy_data['Position']
                    strategy_data['Cumulative_Returns'] = (1 + strategy_data['Strategy_Returns']).cumprod()
                    
                    sharpe_ratio = np.sqrt(252) * strategy_data['Strategy_Returns'].mean() / strategy_data['Strategy_Returns'].std()
                    max_drawdown = (strategy_data['Cumulative_Returns'].cummax() - strategy_data['Cumulative_Returns']).max()
                    max_drawdown_pct = (max_drawdown / strategy_data['Cumulative_Returns'].cummax().max()) * 100

                    buy_trades = pd.DataFrame([t for t in trades if t['Type'] == 'Buy'])
                    sell_trades = pd.DataFrame([t for t in trades if t['Type'] == 'Sell'])
                    total_trades = min(len(buy_trades), len(sell_trades))
                    
                    if total_trades > 0:
                        profitable_trades = sum(1 for t in trades if t['Type'] == 'Sell' and t['Profit_Pct'] > 0)
                        win_rate = (profitable_trades / total_trades) * 100
                    else:
                        win_rate = 0

                    total_return_pct = (strategy_data['Cumulative_Returns'].iloc[-1] - 1) * 100
                    
                    new_row = pd.DataFrame({
                        'Short_MA': [short_ma],
                        'Long_MA': [long_ma],
                        'RSI_Period': [rsi_period],
                        'Bollinger_StdDev': [bb_std_dev],
                        'Total_Return_Pct': [total_return_pct],
                        'Sharpe_Ratio': [sharpe_ratio],
                        'Max_Drawdown_Pct': [max_drawdown_pct],
                        'Win_Rate_Pct': [win_rate],
                        'Total_Trades': [total_trades]
                    })
                    results = pd.concat([results, new_row], ignore_index=True)

    results.dropna(subset=['Sharpe_Ratio'], inplace=True)
    if results.empty:
        logging.info("\n--- No profitable strategies found. ---")
        return
        
    best_strategy = results.loc[results['Sharpe_Ratio'].idxmax()]
    
    logging.info("\n--- Optimization Complete ---")
    logging.info("Best Performing Strategy based on Sharpe Ratio:")
    logging.info(best_strategy.to_string())

    logging.info("\nGenerating chart for the best performing strategy...")
    best_short_ma = int(best_strategy['Short_MA'])
    best_long_ma = int(best_strategy['Long_MA'])
    best_rsi_period = int(best_strategy['RSI_Period'])
    best_bb_std_dev = int(best_strategy['Bollinger_StdDev'])
    
    final_data, _ = apply_vectorized_strategy(data, best_short_ma, best_long_ma, best_rsi_period, best_bb_std_dev)
    
    # 6. Visualize the best strategy
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(final_data['Close'], label='Close Price', alpha=0.5)
    ax1.plot(final_data['Short_MA'], label=f'Short MA ({best_short_ma} days)', color='gold')
    ax1.plot(final_data['Long_MA'], label=f'Long MA ({best_long_ma} days)', color='dodgerblue')
    ax1.plot(final_data['Upper_BB'], label=f'Upper BB ({best_bb_std_dev} StdDev)', linestyle='--', color='purple')
    ax1.plot(final_data['Lower_BB'], label=f'Lower BB ({best_bb_std_dev} StdDev)', linestyle='--', color='purple')
    ax1.plot(final_data.loc[final_data['Signal'] == 1].index, final_data['Close'][final_data['Signal'] == 1], '^', markersize=12, color='lime', label='Buy Signal')
    ax1.plot(final_data.loc[final_data['Signal'] == -1].index, final_data['Close'][final_data['Signal'] == -1], 'v', markersize=12, color='red', label='Sell Signal')
    ax1.set_title(f'{ticker} Best Strategy Performance', fontsize=16)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    

    ax2.plot(final_data['RSI'], label='RSI', color='cyan')
    ax2.axhline(y=50, color='gray', linestyle='--')
    ax2.fill_between(final_data.index, y1=30, y2=70, color='gray', alpha=0.2, label='Normal Range')
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
