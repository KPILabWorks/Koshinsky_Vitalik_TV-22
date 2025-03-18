import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr

# Зчитуємо дані з CSV файлу
data = pd.read_csv("energy_data.csv", parse_dates=["Date"], index_col="Date")

# Визначаємо екстремальні події (припустимо, що вони є у файлі)
events = ["2023-03-15", "2023-07-10", "2023-11-05"]  # Дати екстремальних подій

# Візуалізація змін у споживанні до та після подій
plt.figure(figsize=(12,6))
sns.lineplot(data=data, x=data.index, y="EnergyConsumption", label="Energy Consumption")
for event in events:
    plt.axvline(pd.to_datetime(event), color='red', linestyle='--', label=f'Event: {event}')
plt.title("Energy Consumption Before and After Events")
plt.xlabel("Date")
plt.ylabel("Energy Consumption")
plt.legend()
plt.show()

# Кореляційний аналіз: чи є зміни енергоспоживання зв’язаними зі змінами температури (якщо є у файлі)
if "Temperature" in data.columns:
    correlation, p_value = pearsonr(data["EnergyConsumption"], data["Temperature"])
    print(f"Кореляція між енергоспоживанням та температурою: {correlation:.3f}, p-value: {p_value:.3f}")
else:
    print("Температурні дані відсутні у файлі.")
