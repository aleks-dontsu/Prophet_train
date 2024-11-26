import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import ccxt


def fetch_historical_data(symbol, start_date, end_date):
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date + 'T00:00:00Z')
    all_data = []
    timeframe = '1d'
    limit = 1000

    # Fetch data in chunks until the end date is reached
    while since < exchange.parse8601(end_date + 'T00:00:00Z'):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 86400000  # Move to the next day
            all_data.extend(ohlcv)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    # Create dataframe with the required columns for Prophet
    df = pd.DataFrame(all_data, columns=['ds', 'open', 'high', 'low', 'y', 'volume'])
    df['ds'] = pd.to_datetime(df['ds'], unit='ms')
    df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]
    return df[['ds', 'y']]  # Prophet uses 'ds' and 'y' for training


def train_and_forecast(data, forecast_days):
    model = Prophet()

    # Fit the model using the entire dataset
    model.fit(data)

    # Generate future dates for forecasting
    # future = model.make_future_dataframe(data, periods=forecast_days)
    future = model.make_future_dataframe(periods=forecast_days)

    # Predict the future values
    forecast = model.predict(future)
    return forecast


def compare_forecast_with_actual(forecast, historical):
    # Ensure the forecast and historical data align in length
    min_length = min(len(forecast), len(historical))
    forecast = forecast.iloc[:min_length]
    historical = historical.iloc[:min_length]

    # Add the actual values to the forecast
    forecast['actual'] = historical['y'].values
    forecast['error'] = forecast['actual'] - forecast['yhat']

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(forecast['actual'], forecast['yhat'])
    return forecast, mae


def plot_forecast_with_actual(historical, forecast, symbol, save_path=None):
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(historical['ds'], historical['y'], label="Actual Prices", color='blue')

    # Plot forecast data
    plt.plot(forecast['ds'], forecast['yhat'], label="Forecast Prices", color='orange')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2,
                     label="Confidence Interval")

    plt.title(f"{symbol} Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Cryptocurrency Forecasting Tool')
    parser.add_argument('--symbol', type=str, required=True, help='Cryptocurrency pair (e.g., BTC/USDT)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--forecast_days', type=int, required=True, help='Number of days to forecast')
    parser.add_argument('--save_csv', type=bool, default=True, help='Save CSV files locally (True/False)')
    args = parser.parse_args()

    print("Fetching historical data...")
    historical_data = fetch_historical_data(args.symbol, args.start_date, args.end_date)

    # Save historical data to CSV
    if args.save_csv:
        csv_folder = 'csv_files'
        os.makedirs(csv_folder, exist_ok=True)
        historical_path = os.path.join(csv_folder, 'crypto_historical_data.csv')
        historical_data.to_csv(historical_path, index=False)
        print(f"Historical data saved at: {historical_path}")

    print("Training and forecasting...")
    forecast = train_and_forecast(historical_data, args.forecast_days)

    # Save forecast data to CSV
    if args.save_csv:
        forecast_path = os.path.join(csv_folder, 'crypto_price_forecast.csv')
        forecast.to_csv(forecast_path, index=False)
        print(f"Forecast data saved at: {forecast_path}")

    print("Comparing forecast with actual data...")
    forecast_with_actual, mae = compare_forecast_with_actual(forecast, historical_data)
    print(f"Mean Absolute Error: {mae}")

    # Save forecast with actual data to CSV
    if args.save_csv:
        combined_path = os.path.join(csv_folder, 'crypto_forecast_with_actual.csv')
        forecast_with_actual.to_csv(combined_path, index=False)
        print(f"Forecast with actual data saved at: {combined_path}")

    print("Plotting forecast...")
    plot_forecast_with_actual(historical_data, forecast, args.symbol, save_path='crypto_forecast_plot.png')


if __name__ == '__main__':
    main()
