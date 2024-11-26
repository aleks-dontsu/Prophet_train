import ccxt
import pandas as pd
from datetime import datetime
from prophet import Prophet
import matplotlib.pyplot as plt

# === 1. Сбор данных через Binance API ===
def fetch_historical_data():
    binance = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'
    since = binance.parse8601('2014-11-22T00:00:00Z')  # 10 лет назад

    # Скачивание данных
    print("Скачивание данных с Binance...")
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since)

    # Преобразование в DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Сохранение в файл
    df.to_csv('btc_historical_data.csv', index=False)
    print("Данные сохранены в файл: btc_historical_data.csv")
    return df

# === 2. Подготовка данных для Prophet ===
def prepare_data_for_prophet(df):
    df_prophet = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    return df_prophet

# === 3. Обучение модели Prophet и прогнозирование ===
def train_and_forecast(df_prophet, periods=30):
    print("Обучение модели Prophet...")
    model = Prophet()
    model.fit(df_prophet)

    print("Создание прогноза...")
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Сохранение прогноза
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('btc_price_forecast.csv', index=False)
    print("Прогноз сохранён в файл: btc_price_forecast.csv")
    return forecast, model

# === 4. Сравнение прогноза с фактическими данными ===
def compare_forecast_with_actual(forecast, historical):
    # Сопоставление только тех дат, которые есть в исторических данных
    merged = pd.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        historical[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'actual'}),
        on='ds',
        how='inner'  # Оставляем только общие даты
    )

    # Расчёт средней абсолютной ошибки (MAE)
    mae = abs(merged['actual'] - merged['yhat']).mean()
    print(f"Средняя абсолютная ошибка (MAE): {mae}")

    # Сохранение результата в файл
    merged.to_csv('btc_price_forecast_with_actual.csv', index=False)
    print("Файл с прогнозом и фактическими данными сохранён: btc_price_forecast_with_actual.csv")
    return merged, mae

# === 5. Визуализация данных ===
def plot_forecast(forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'], forecast['actual'], label='Фактические данные', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Прогноз', color='orange')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2, label='Доверительный интервал')
    plt.legend()
    plt.title('Сравнение прогноза и фактических данных')
    plt.xlabel('Дата')
    plt.ylabel('Цена BTC/USDT')
    plt.grid()
    plt.show()

# === Основной блок выполнения ===
if __name__ == '__main__':
    # 1. Сбор данных
    historical_data = fetch_historical_data()

    # 2. Подготовка данных
    prophet_data = prepare_data_for_prophet(historical_data)

    # 3. Обучение модели и прогнозирование
    forecast, model = train_and_forecast(prophet_data, periods=30)

    # 4. Сравнение с фактическими данными
    forecast_with_actual, mae = compare_forecast_with_actual(forecast, historical_data)

    # 5. Визуализация
    plot_forecast(forecast_with_actual)
