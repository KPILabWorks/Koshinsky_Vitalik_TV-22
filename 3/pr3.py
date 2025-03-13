import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Завантаження даних
file_path = "time_series_15min_singleindex.csv"
df = pd.read_csv(file_path, parse_dates=['utc_timestamp'], index_col='utc_timestamp')

df = df[['AT_load_actual_entsoe_transparency']].dropna()
df = df.resample('D').mean()  # Агрегація до денного рівня

# Декомпозиція часових рядів
decomposition = sm.tsa.seasonal_decompose(df, model='additive', period=365)


# Візуалізація компонентів з кольорами та мітками років
def plot_decomposition(decomposition):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title='Спостережувані дані', color='blue')
    decomposition.trend.plot(ax=axes[1], title='Тренд', color='red')
    decomposition.seasonal.plot(ax=axes[2], title='Сезонність', color='green')
    decomposition.resid.plot(ax=axes[3], title='Залишки', color='purple')

    # Додаємо мітки років
    axes[3].set_xticks(pd.date_range(df.index.min(), df.index.max(), freq='YS'))
    axes[3].set_xticklabels(pd.date_range(df.index.min(), df.index.max(), freq='YS').year, rotation=45)

    plt.tight_layout()
    plt.show()


plot_decomposition(decomposition)

# Оцінка моделі
from sklearn.metrics import mean_absolute_error

# Простий прогноз на основі тренду (можна замінити на більш складні моделі)
trend = decomposition.trend.dropna()
predictions = trend.shift(1)
actual = trend.loc[predictions.index]

# Видалення NaN значень перед обчисленням MAE
valid_idx = actual.notna() & predictions.notna()
actual_clean = actual[valid_idx]
predictions_clean = predictions[valid_idx]

mae = mean_absolute_error(actual_clean, predictions_clean)
print(f"Mean Absolute Error: {mae:.2f}")
