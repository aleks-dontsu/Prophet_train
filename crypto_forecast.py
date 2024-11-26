import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import ccxt

def fetch_historical_data(symbol, start_date, end_date):
    exchange = ccxt.binance()
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_timestamp = exchange.parse8601(f"{end_date}T00:00:00Z")
    timeframe = '1d'

    all_data = []
    while since < end_timestamp:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 86400000

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(all_data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns={'timestamp': 'date'}, inplace=True)
    return df

def train_and_forecast(data, forecast_days):
    df = data[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast

def compare_forecast_with_actual(forecast, historical):
    forecast['actual'] = historical['close'].values[:len(forecast)]
    forecast_with_actual = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'actual']]
    forecast_with_actual['error'] = forecast_with_actual['actual'] - forecast_with_actual['yhat']
    mae = mean_absolute_error(forecast_with_actual['actual'], forecast_with_actual['yhat'])
    return forecast_with_actual, mae

def plot_forecast(forecast_with_actual, symbol, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_with_actual['ds'], forecast_with_actual['actual'], label='Actual Prices', color='blue')
    plt.plot(forecast_with_actual['ds'], forecast_with_actual['yhat'], label='Forecast Prices', color='orange')
    plt.fill_between(
        forecast_with_actual['ds'],
        forecast_with_actual['yhat_lower'],
        forecast_with_actual['yhat_upper'],
        color='orange', alpha=0.2, label='Confidence Interval'
    )
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{symbol} Price Forecast')
    plt.legend()

    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, f"{symbol.replace('/', '_')}_forecast_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved at: {plot_path}")

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

    if args.save_csv:
        csv_folder = 'csv_files'
        os.makedirs(csv_folder, exist_ok=True)
        historical_path = os.path.join(csv_folder, 'crypto_historical_data.csv')
        historical_data.to_csv(historical_path, index=False)
        print(f"Historical data saved at: {historical_path}")

    print("Training and forecasting...")
    forecast = train_and_forecast(historical_data, args.forecast_days)

    if args.save_csv:
        forecast_path = os.path.join(csv_folder, 'crypto_price_forecast.csv')
        forecast.to_csv(forecast_path, index=False)
        print(f"Forecast data saved at: {forecast_path}")

    print("Comparing forecast with actual data...")
    forecast_with_actual, mae = compare_forecast_with_actual(forecast, historical_data)
    print(f"Mean Absolute Error: {mae}")

    if args.save_csv:
        combined_path = os.path.join(csv_folder, 'crypto_forecast_with_actual.csv')
        forecast_with_actual.to_csv(combined_path, index=False)
        print(f"Forecast with actual data saved at: {combined_path}")

    print("Plotting forecast...")
    plot_forecast(forecast_with_actual, args.symbol, save_path='plots')

if __name__ == '__main__':
    main()
