# Cryptocurrency Forecasting Using Prophet

This project enables cryptocurrency price forecasting using historical data and the [Prophet](https://facebook.github.io/prophet/) library.  
It collects data from Binance, trains a forecasting model, and predicts prices for the desired cryptocurrency.

---

## **Features**
- Fetch historical price data for any cryptocurrency pair (e.g., BTC/USDT) using [ccxt](https://github.com/ccxt/ccxt).
- Train a forecasting model using [Prophet](https://facebook.github.io/prophet/).
- Save historical data and forecasts to `.csv` files.
- Visualize forecast data alongside actual historical data with confidence intervals.
- Easily customize the forecast period and date range.

---

## **Setup Instructions**

### 1. **Install Dependencies**
Install the required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## **Run app**

```bash
python crypto_forecast.py --symbol BTC/USDT --start_date 2015-01-01 --end_date 2023-12-31 --forecast_days 365
```

## **File Outputs**

- crypto_historical_data.csv: Historical OHLCV data.
- crypto_price_forecast.csv: Forecast data for the specified period.
- crypto_forecast_with_actual.csv: Combined file with forecast and actual prices.
- [symbol]_forecast_plot.png: A plot of forecast vs. actual prices (e.g., BTC_USDT_forecast_plot.png).