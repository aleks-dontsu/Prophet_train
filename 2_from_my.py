import ccxt
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import time


# 1. Загрузка исторических данных
def fetch_historical_data():
    binance = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'
    since = binance.parse8601('2015-01-01T00:00:00Z')  # Начало 2015 года

    all_data = []  # Для накопления данных
    limit = 1000   # Максимальное количество свечей за один запрос
    current_since = since  # Текущая начальная точка

    print("Скачивание данных с Binance...")

    while True:
        # Запрос данных
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
            break

        if not ohlcv:
            print("Данные полностью загружены.")
            break

        # Добавление данных в общий список
        all_data.extend(ohlcv)

        # Обновление начальной точки
        current_since = ohlcv[-1][0] + 1  # Следующая свеча после последней

        # Задержка для предотвращения блокировки API
        time.sleep(1)

    # Преобразование данных в DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Проверка диапазона данных
    print(f"Первые данные: {df['timestamp'].min()}, Последние данные: {df['timestamp'].max()}")

    # Сохранение в файл
    df.to_csv('btc_historical_data.csv', index=False)
    print("Данные сохранены в файл: btc_historical_data.csv")

    return df

# 2. Подготовка данных для Prophet
def prepare_data_for_prophet(df):
    df_prophet = df[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    return df_prophet

# 3. Обучение модели Prophet и создание прогноза
def train_and_forecast(df_prophet, periods):
    print("Обучение модели Prophet...")
    model = Prophet()
    model.fit(df_prophet)

    print("Создание прогноза...")
    future = model.make_future_dataframe(periods=periods)  # Прогноз на заданное число дней вперёд
    forecast = model.predict(future)

    # Сохранение прогноза
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('btc_price_forecast.csv', index=False)
    print("Прогноз сохранён в файл: btc_price_forecast.csv")
    return forecast, model

# 4. Объединение прогноза с фактическими данными
def compare_forecast_with_actual(forecast, historical_data):
    print("Объединение прогноза с фактическими данными...")
    historical_data['ds'] = pd.to_datetime(historical_data['timestamp'])
    historical_data = historical_data[['ds', 'close']]  # Убедимся, что используем только нужные столбцы

    # Объединение прогноза с фактическими данными
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    combined = pd.merge(forecast, historical_data, on='ds', how='left')
    combined.rename(columns={'close': 'actual'}, inplace=True)

    # Сохранение в файл
    combined.to_csv('btc_price_forecast_with_actual.csv', index=False)
    print("Прогноз и фактические данные сохранены в файл: btc_price_forecast_with_actual.csv")

    # Вычисление MAE (средней абсолютной ошибки)
    valid_data = combined.dropna(subset=['actual'])
    mae = mean_absolute_error(valid_data['actual'], valid_data['yhat'])
    print(f"Средняя абсолютная ошибка (MAE): {mae}")
    return combined, mae

# 5. Построение графика с фактическими данными и прогнозом
def plot_forecast_with_actual(combined):
    plt.figure(figsize=(14, 7))
    plt.plot(combined['ds'], combined['yhat'], label='Прогноз', color='orange')
    plt.fill_between(combined['ds'], combined['yhat_lower'], combined['yhat_upper'], color='gray', alpha=0.2, label='Доверительный интервал')
    plt.plot(combined['ds'], combined['actual'], label='Фактические данные', color='blue', alpha=0.6)
    plt.axvline(pd.Timestamp.today(), color='red', linestyle='--', label='Сегодня')
    plt.legend()
    plt.title('Прогноз и фактические данные BTC/USDT')
    plt.xlabel('Дата')
    plt.ylabel('Цена BTC/USDT')
    plt.grid()

    # Сохранение графика
    filename = 'btc_forecast_with_actual_plot.png'
    plt.savefig(filename, format='png')
    print(f"График сохранён в файл: {filename}")

    # Показать график
    plt.show()

# Основной процесс
def main():
    # 1. Скачиваем исторические данные
    historical_data = fetch_historical_data()

    # 2. Готовим данные для Prophet
    df_prophet = prepare_data_for_prophet(historical_data)

    # 3. Создаём и обучаем модель, прогнозируем на 2 года вперёд
    forecast, model = train_and_forecast(df_prophet, periods=730)

    # 4. Сравниваем прогноз с фактическими данными и сохраняем результат
    combined, mae = compare_forecast_with_actual(forecast, historical_data)

    # 5. Строим и сохраняем график
    plot_forecast_with_actual(combined)

if __name__ == '__main__':
    main()