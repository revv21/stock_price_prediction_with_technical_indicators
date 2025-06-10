import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def fetch_nq024_data(period="1mo", interval="1h"):
    """
    Fetch NQ024 (Nasdaq futures) data and format it according to the specified structure
    
    Parameters:
    period: str - Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    interval: str - Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
    pandas.DataFrame - Formatted data
    """
    
    # Try different ticker symbols for Nasdaq futures
    tickers = ["NQ=F", "^IXIC", "QQQ"]  # NQ futures, Nasdaq index, Nasdaq ETF
    
    for ticker in tickers:
        try:
            print(f"Trying ticker: {ticker}")
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data retrieved for {ticker}. Trying next ticker...")
                continue
            
            print(f"Successfully retrieved {len(data)} records from {ticker}")
            print(f"Data columns: {list(data.columns)}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            # Reset index to get datetime as a column
            data = data.reset_index()
            
            # Handle different datetime column names
            if 'Datetime' in data.columns:
                datetime_col = 'Datetime'
            elif 'Date' in data.columns:
                datetime_col = 'Date'
            else:
                print(f"No datetime column found. Available columns: {list(data.columns)}")
                continue
            
            # Create the formatted dataframe
            formatted_data = pd.DataFrame()
            
            # Format datetime to match the pattern in your image
            formatted_data['Symbol'] = 'NQ024'
            formatted_data['Time'] = data[datetime_col].dt.strftime('%m/%d/%Y %H:%M')
            formatted_data['Open'] = data['Open'].round(2)
            formatted_data['High'] = data['High'].round(2)
            formatted_data['Low'] = data['Low'].round(2)
            formatted_data['Last'] = data['Close'].round(2)  # Last price is typically the close price
            
            # Calculate Change (difference from previous close)
            formatted_data['Change'] = (data['Close'] - data['Close'].shift(1)).round(2)
            
            # Calculate %Chg (percentage change)
            formatted_data['%Chg'] = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100).round(2)
            
            # Volume - handle if volume is not available
            if 'Volume' in data.columns:
                formatted_data['Volume'] = data['Volume'].fillna(0).astype(int)
            else:
                formatted_data['Volume'] = 0
            
            # Add "Open Int" column (Open Interest - not available in basic yfinance data)
            formatted_data['Open Int'] = 'N/A'
            
            # For the first row, set Change and %Chg to 0 instead of removing it
            formatted_data.loc[0, 'Change'] = 0.0
            formatted_data.loc[0, '%Chg'] = 0.0
            
            print(f"Formatted data shape: {formatted_data.shape}")
            
            if len(formatted_data) > 0:
                return formatted_data
            else:
                print(f"No data after formatting for {ticker}")
                continue
                
        except Exception as e:
            print(f"Error fetching data from {ticker}: {e}")
            continue
    
    print("Failed to retrieve data from all ticker symbols")
    return None

def save_to_csv(data, filename=None):
    """
    Save the formatted data to a CSV file
    
    Parameters:
    data: pandas.DataFrame - The formatted data
    filename: str - Optional filename. If None, uses timestamp
    """
    if data is None:
        print("No data to save.")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nq024.csv"
    
    # Create header row to match your format
    header = "Symbol,Time,Open,High,Low,Last,Change,%Chg,Volume,\"Open Int\""
    
    try:
        data.to_csv(filename, index=False, header=False)
        
        # Read the file and add the custom header
        with open(filename, 'r') as f:
            content = f.read()
        
        with open(filename, 'w') as f:
            f.write(header + '\n' + content)
        
        print(f"Data saved to {filename}")
        
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    """
    Main function to fetch and save NQ024 data
    """
    print("Fetching NQ024 (Nasdaq Futures) data...")
    
    # Try different periods and intervals if the first one fails
    configs = [
        {"period": "5d", "interval": "1h"},
        {"period": "1mo", "interval": "1d"},
        {"period": "5d", "interval": "1d"},
        {"period": "1d", "interval": "5m"}
    ]
    
    data = None
    for config in configs:
        print(f"\nTrying period='{config['period']}', interval='{config['interval']}'")
        data = fetch_nq024_data(period=config["period"], interval=config["interval"])
        if data is not None and len(data) > 0:
            break
    
    if data is not None and len(data) > 0:
        print(f"\nSuccessfully retrieved {len(data)} records")
        print("\nFirst few records:")
        print(data.head())
        
        # Save to CSV
        save_to_csv(data)
        
        # Display statistics with proper error handling
        if len(data) > 0:
            print(f"\nData Summary:")
            print(f"Date range: {data['Time'].iloc[0]} to {data['Time'].iloc[-1]}")
            print(f"Price range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}")
            print(f"Latest price: ${data['Last'].iloc[-1]:.2f}")
            
            # Check if we have valid change data
            if not pd.isna(data['Change'].iloc[-1]):
                print(f"Latest change: {data['Change'].iloc[-1]:.2f} ({data['%Chg'].iloc[-1]:.2f}%)")
            else:
                print("Latest change: N/A (insufficient data)")
        
    else:
        print("Failed to retrieve data with any configuration.")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try running the script during market hours")
        print("3. Yahoo Finance might be temporarily unavailable")
        print("4. Try using a different data source or API")

# Alternative function for different time periods
def fetch_custom_period(days=30, interval="1h"):
    """
    Fetch data for a custom number of days
    
    Parameters:
    days: int - Number of days to fetch
    interval: str - Data interval
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    ticker = "NQ=F"
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            print("No data retrieved for custom period.")
            return None
        
        # Format the data (same as above)
        data = data.reset_index()
        formatted_data = pd.DataFrame()
        
        formatted_data['Symbol'] = 'NQ024'
        formatted_data['Time'] = data['Datetime'].dt.strftime('%m/%d/%Y %H:%M')
        formatted_data['Open'] = data['Open'].round(2)
        formatted_data['High'] = data['High'].round(2)
        formatted_data['Low'] = data['Low'].round(2)
        formatted_data['Last'] = data['Close'].round(2)
        formatted_data['Change'] = (data['Close'] - data['Close'].shift(1)).round(2)
        formatted_data['%Chg'] = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100).round(2)
        formatted_data['Volume'] = data['Volume'].fillna(0).astype(int)
        formatted_data['Open Int'] = 'N/A'
        
        formatted_data = formatted_data.dropna()
        
        return formatted_data
        
    except Exception as e:
        print(f"Error fetching custom period data: {e}")
        return None

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import yfinance
        import pandas
        import numpy
    except ImportError:
        print("Please install required packages:")
        print("pip install yfinance pandas numpy")
        exit(1)
    
    main()
    
    # Example of fetching data for last 7 days with 30-minute intervals
    print("\n" + "="*50)
    print("Fetching last 7 days data with 30-minute intervals...")
    custom_data = fetch_custom_period(days=7, interval="30m")
    if custom_data is not None:
        save_to_csv(custom_data, "nq024_last_7days.csv")