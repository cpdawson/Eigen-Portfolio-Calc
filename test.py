import yfinance as yf
import pandas as pd


def get_prices_on_dates(ticker, date):
    """
    Fetch the closing price of a ticker on a specified date and one month later.

    Args:
    ticker (str): The stock ticker symbol e.g., 'AAPL'.
    date (str): The specific date in format 'YYYY-MM-DD'.

    Returns:
    tuple: A tuple containing the closing prices on the specified date and one month later.
    """
    # Convert the input date to datetime
    date = pd.to_datetime(date)

    # Define the start and end date for the download window
    start_date = date - pd.DateOffset(days=10)  # Start a bit earlier to ensure we get the exact date
    end_date = date + pd.DateOffset(days=40)  # End a bit later to ensure coverage

    # Fetch data from Yahoo Finance
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)

    if data.empty:
        return (None, None)  # Return None if no data was retrieved

    # Find the closest prices to the specified date and one month later
    price_on_date = data['Close'].asof(date)
    price_one_month_later = data['Close'].asof(date + pd.DateOffset(months=1))

    return (price_on_date, price_one_month_later)

if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    date = '2024-05-15'
    prices = get_prices_on_dates(ticker, date)
    print(f"Price on {date}: {prices[0]}, Price one month later: {prices[1]}")
