import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from pandas.tseries.offsets import MonthEnd


def fetch_data(tickers, start_date, end_date):
    """ Fetches adjusted closing prices for given tickers between dates. """
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data.dropna(how='all', axis=0)  # Drop days where all tickers are NA
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()


def get_prices_on_dates(ticker, date):
    """ Fetches the price of a stock on a specific date and a month later. """
    date = pd.to_datetime(date)
    initial_date = date - pd.DateOffset(days=10)
    final_date = date + pd.DateOffset(days=40)
    try:
        data = yf.download(ticker, start=initial_date.strftime('%Y-%m-%d'), end=final_date.strftime('%Y-%m-%d'),
                           progress=False)
        if data.empty:
            return None, None
        price_on_date = data['Close'].asof(date)
        price_one_month_later = data['Close'].asof(date + pd.DateOffset(months=1))
        return price_on_date, price_one_month_later
    except Exception as e:
        print(f"Failed to fetch prices for {ticker} due to {e}")
        return None, None


def construct_eigenportfolios(data, variance_threshold=0.95):
    """ Calculates eigenportfolios using PCA on the return data. """
    returns = data.pct_change().dropna()
    returns_standardized = (returns - returns.mean()) / returns.std()
    n_components = min(len(data.columns), len(returns), int(variance_threshold * len(data.columns)))
    pca = PCA(n_components=n_components)
    pca.fit(returns_standardized)
    weights = np.zeros(len(data.columns))
    if len(pca.components_) > 0:
        weights[:len(pca.components_[0])] = pca.components_[0]
    weights /= np.sum(np.abs(weights))  # Normalize weights
    return weights


def calculate_portfolio_weights(tickers, start_date, end_date):
    """ Calculates and stores portfolio weights for a list of tickers over specified date ranges. """
    results_df = pd.DataFrame()
    current_date = pd.to_datetime(start_date)

    while current_date < pd.to_datetime(end_date):
        month_end = current_date + MonthEnd(1) - pd.DateOffset(days=1)
        data = fetch_data(tickers, (month_end - MonthEnd(1) + pd.DateOffset(days=1)).strftime('%Y-%m-%d'),
                          month_end.strftime('%Y-%m-%d'))

        if not data.empty:
            # Ensure the data includes enough dates (at least for a meaningful analysis)
            weights = construct_eigenportfolios(data)
            month_data = {f'Weight_{ticker}': weights[i] if i < len(weights) else 0 for i, ticker in
                          enumerate(data.columns)}
            month_weights_df = pd.DataFrame([month_data], index=[month_end])
            month_weights_df.index.name = 'Date'  # Naming the index
            results_df = pd.concat([results_df, month_weights_df])

        else:
            print(f"No data found for period ending {month_end}")

        current_date = month_end + pd.DateOffset(days=1)

    return results_df

# Example usage
all_tickers = [
    "AAPL", "GOOG", "GOOGL", "BRK.B", "LLY", "AVGO", "JPM", "V", "WMT", "XOM",
    "PG", "ORCL", "COST", "JNJ", "HD", "BAC", "ABBV", "NFLX", "CVX", "KO",
    "QCOM", "ADBE", "PEP", "CRM", "TMO", "LIN", "TMUS", "WFC", "AMAT", "DHR",
    "MCD", "ABT", "ACN", "INTU", "AXP", "AMGN", "PM", "MS", "ISRG", "NEE",
    "BX", "GS", "NKE", "RTX", "HON", "UNP", "LRCX", "SCHW", "SYK", "BKNG",
    "ETN", "COP", "LOW", "ELV", "VRTX", "TJX", "PGR", "UPS", "ADI", "BLK",
    "REGN", "KLAC", "PLD", "CB", "DE", "ANET", "PANW", "CI", "CMG", "HCA",
    "FI", "SO", "CDNS", "APH", "WM", "GD", "DUK", "ICE", "CL", "MCK", "CVS",
    "SHW", "MCO", "EQIX", "TDG", "CTAS", "FCX", "NXPI", "ECL", "EOG", "CEG",
    "BDX", "TGT", "PH", "CSX", "NOC", "MSI", "WELL", "APD", "FDX", "PNC",
    "MPC", "RSG", "ORLY", "ROP", "PSX", "AJG", "MRNA", "GM", "OXY", "EW",
    "HLT", "CPRT", "COF", "WMB", "MNST", "PSA", "NSC", "SMCI", "VLO", "MCHP",
    "ADSK", "MET", "AIG", "ROST", "AZO", "DLR", "SRE", "NEM", "DHI", "JCI",
    "G", "FTNT", "COR", "KDP", "OKE", "TEL", "BK", "HUM", "LEN", "AMP", "IDXX",
    "ALL", "LHX", "URI", "PWR", "LULU", "KHC", "OTIS", "MPWR", "IQV", "AME",
    "YUM", "RCL", "MSCI", "VRSK", "NUE", "IR", "CNC", "KR", "ACGL", "EA",
    "PEG", "CTVA", "MLM", "GEHC", "HPQ", "NDAQ", "BIIB", "IT", "XYL", "VMC",
    "DD", "FANG", "HWM", "GLW", "RMD", "BKR", "ON", "TSCO", "CSGP", "CDW",
    "EFX", "PPG", "HIG", "FSLR", "DVN", "WAB", "EIX", "TTWO", "CBRE", "WTW",
    "DECK", "IRM", "BRO", "FTV", "WDC", "AWK", "VLTO", "CAH"
]
required_tickers = ["AAPL", "GOOGL", "LLY", "AVGO", "JPM", "V", "WMT", "XOM"]
random_tickers = np.random.choice(all_tickers, 10, replace=False)
random_tickers = random_tickers.tolist()
tickers = required_tickers + random_tickers


start_date = '2023-01-01'
end_date = '2024-06-01'
portfolio_results = calculate_portfolio_weights(tickers, start_date, end_date)
print(portfolio_results)

for ticker in tickers:
    changes_list = []
    profit_list = []
    for date in portfolio_results.index[:-1]:
        prices = get_prices_on_dates(ticker, date)
        if prices[0] is None or prices[1] is None:
            pct_change = None
        else:
            pct_change = (prices[1] - prices[0]) / prices[0] if prices[0] != 0 else None
        changes_list.append(pct_change)
    changes_list.append(None)
    portfolio_results[f'Pct_change_{ticker}'] = changes_list

def calculate_profit(portfolio_results):
    """ Calculate the monthly profit for each ticker in the portfolio. """
    profit_df = pd.DataFrame(index=portfolio_results.index)
    for ticker in portfolio_results.filter(like='Weight_').columns:
        ticker_name = ticker.split('_')[-1]
        weight_column = f'Weight_{ticker_name}'
        change_column = f'Pct_change_{ticker_name}'

        # Initialize an empty list to store profit calculations for each month
        monthly_profits = []

        # Iterate over each row by index to apply specific profit calculations
        for idx in portfolio_results.index[:-1]:  # Avoid the last index since it might not have a corresponding change value
            weight = portfolio_results.at[idx, weight_column]
            pct_change = portfolio_results.at[idx, change_column]
            if pd.isna(pct_change):
                # Append NaN if the pct_change is not available (e.g., latest month)
                monthly_profits.append(np.nan)
            else:
                # Calculate profit for positions
                profit = weight * pct_change
                monthly_profits.append(profit)

        # Ensure that the list of monthly profits matches the index length exactly
        if len(monthly_profits) < len(portfolio_results.index):
            # Append NaN for the last month if necessary
            monthly_profits.append(np.nan)

        # Assign the list of monthly profits to the appropriate column in the DataFrame
        profit_df[f'Profit_{ticker_name}'] = monthly_profits

    # Sum up all profits for a total monthly portfolio profit
    profit_df['Total_Profit'] = profit_df.sum(axis=1)
    return profit_df



# Assuming 'portfolio_results' is your DataFrame containing the weights and pct changes
profit_results = calculate_profit(portfolio_results)
print(profit_results)
final_df = pd.concat([portfolio_results, profit_results], axis=1)
# Optionally, save to Excel
final_df.to_excel('portfolio_monthly_profits.xlsx')


