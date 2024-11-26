import ccxt
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import argparse
import time


# 1. Fetch historical data
def fetch_historical_data(symbol, start_date, end_date):
    binance = ccxt.binance()
    timeframe = '1d'
    since = binance.parse8601(f'{start_date}T00:00:00Z')
    end_timestamp = binance.parse8601(f'{end_date}T00:00:00Z')

    all_data = []
    limit = 1000
    current_since = since

    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")

    while True:
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

        if not ohlcv:
            break

        # Append data
        all_data.extend(ohlcv)

        # Check if end date is reached
        if ohlcv[-1][0] >= end_timestamp:
            break

        # Update since timestamp
        current_since = ohlcv[-1][0] + 1
        time.sleep(1)

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Filter by end date
    df = df[df['timestamp'] <= pd.to_datetime(end_date)]

    print(f"Fetched {len(df)} records.")
    df.to_csv('crypto_historical_data.csv', index=False)
    print("Data saved to file: crypto_historical_data.csv")

    return df


# 2. Prepare data for Prophet
def prepare_data_for_prophet(df):
    df_prophet = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    return df_prophet


# 3. Train and forecast using Prophet
def train_and_forecast(df_prophet, forecast_days):
    print(f"Training Prophet model and forecasting for the next {forecast_days} days...")
    model = Prophet()
    model.fit(df_prophet)

    # Create forecast
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Save forecast
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('crypto_price_forecast.csv', index=False)
    print("Forecast saved to file: crypto_price_forecast.csv")

    return forecast, model


# 4. Merge forecast with actual data
def compare_forecast_with_actual(forecast, historical_data):
    print("Merging forecast with actual data...")
    historical_data['ds'] = pd.to_datetime(historical_data['timestamp'])
    historical_data = historical_data[['ds', 'close']]

    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    combined = pd.merge(forecast, historical_data, on='ds', how='left')
    combined.rename(columns={'close': 'actual'}, inplace=True)

    combined.to_csv('crypto_forecast_with_actual.csv', index=False)
    print("Forecast and actual data saved to file: crypto_forecast_with_actual.csv")

    # Calculate MAE
    valid_data = combined.dropna(subset=['actual'])
    mae = mean_absolute_error(valid_data['actual'], valid_data['yhat'])
    print(f"Mean Absolute Error (MAE): {mae}")

    return combined, mae


# 5. Plot forecast and actual data
def plot_forecast_with_actual(combined, symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(combined['ds'], combined['yhat'], label='Forecast', color='orange')
    plt.fill_between(combined['ds'], combined['yhat_lower'], combined['yhat_upper'], color='gray', alpha=0.2, label='Confidence Interval')
    plt.plot(combined['ds'], combined['actual'], label='Actual', color='blue', alpha=0.6)
    plt.axvline(pd.Timestamp.today(), color='red', linestyle='--', label='Today')
    plt.legend()
    plt.title(f'Forecast vs Actual for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid()

    filename = f'{symbol.replace("/", "_")}_forecast_plot.png'
    plt.savefig(filename, format='png')
    print(f"Plot saved to file: {filename}")

    plt.show()


# Main process
def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency forecast using Prophet.")
    parser.add_argument('--symbol', type=str, required=True, help="Cryptocurrency pair (e.g., BTC/USDT).")
    parser.add_argument('--start_date', type=str, required=True, help="Start date for data collection (e.g., 2015-01-01).")
    parser.add_argument('--end_date', type=str, required=True, help="End date for data collection (e.g., 2023-12-31).")
    parser.add_argument('--forecast_days', type=int, required=True, help="Number of days to forecast.")

    args = parser.parse_args()

    # Fetch historical data
    historical_data = fetch_historical_data(args.symbol, args.start_date, args.end_date)

    # Prepare data for Prophet
    df_prophet = prepare_data_for_prophet(historical_data)

    # Train and forecast
    forecast, model = train_and_forecast(df_prophet, args.forecast_days)

    # Merge forecast with actual data
    combined, mae = compare_forecast_with_actual(forecast, historical_data)

    # Plot forecast and actual data
    plot_forecast_with_actual(combined, args.symbol)


if __name__ == '__main__':
    main()
