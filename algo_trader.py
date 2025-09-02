import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def backtest_moving_average_crossover(ticker, short_window, long_window):
    """
    Backtests a moving average crossover strategy on a given stock.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
        short_window (int): The number of days for the short-term moving average.
        long_window (int): The number of days for the long-term moving average.
    """
    print(f"Running backtest for {ticker}...")

    # 1. Download historical data
    # We download data for the last 5 years.
    try:
        data = yf.download(ticker, period="10y")
        if data.empty:
            print(f"Error: No data found for ticker {ticker}. Please check the symbol.")
            return
    except Exception as e:
        print(f"An error occurred while downloading data: {e}")
        return

    # 2. Calculate moving averages
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

    # Drop any rows with NaN values that result from the rolling mean calculation
    data.dropna(inplace=True)

    # 3. Generate trading signals
    # A new column 'Signal' is created, initialized with 0 (no signal).
    # A '1' indicates a buy signal, a '-1' indicates a sell signal.
    data['Signal'] = 0.0

    # A buy signal is generated when the short MA crosses above the long MA
    data['Signal'][short_window:] = (data['Short_MA'][short_window:] > data['Long_MA'][short_window:]).astype(int)

    # We take the difference to find the exact moments of the crossover
    # 1 indicates a buy event (0 -> 1)
    # -1 indicates a sell event (1 -> 0)
    data['Positions'] = data['Signal'].diff()

    # 4. Calculate performance
    # Backtest logic:
    # We'll calculate the hypothetical returns if we followed the signals.
    # We assume we buy 1 share on a 'buy' signal and sell 1 share on a 'sell' signal.
    initial_capital = 10000.0  # Starting capital for our simulation
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    positions[ticker] = 100 * data['Signal'] # Number of shares to hold (100 in this case)

    # Calculate portfolio value over time
    portfolio = positions.multiply(data['Close'], axis=0)
    pos_diff = positions.diff()
    
    portfolio['Holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
    portfolio['Cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()
    portfolio['Total'] = portfolio['Holdings'] + portfolio['Cash']

    # 5. Visualize the results
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 8))
    
    # Plotting the stock's closing price
    plt.plot(data['Close'], label='Close Price', alpha=0.5)
    
    # Plotting the moving averages
    plt.plot(data['Short_MA'], label=f'Short MA ({short_window} days)', color='gold')
    plt.plot(data['Long_MA'], label=f'Long MA ({long_window} days)', color='dodgerblue')

    # Plotting buy signals
    plt.plot(data.loc[data['Positions'] == 1.0].index, 
             data['Short_MA'][data['Positions'] == 1.0],
             '^', markersize=10, color='lime', label='Buy Signal')

    # Plotting sell signals
    plt.plot(data.loc[data['Positions'] == -1.0].index,
             data['Short_MA'][data['Positions'] == -1.0],
             'v', markersize=10, color='red', label='Sell Signal')

    plt.title(f'{ticker} Moving Average Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    # Print the final performance
    final_value = portfolio['Total'].iloc[-1]
    net_return = ((final_value - initial_capital) / initial_capital) * 100
    
    print("\n--- Backtest Results ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Net Return: {net_return:.2f}%")
    
# --- Main execution ---
if __name__ == '__main__':
    # You can change these parameters to test different stocks and strategies
    STOCK_TICKER = 'MSFT' # Microsoft
    SHORT_TERM = 50
    LONG_TERM = 300

    backtest_moving_average_crossover(STOCK_TICKER, SHORT_TERM, LONG_TERM)
