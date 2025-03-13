import pandas as pd
import numpy as np
import time

# Генеруємо великий DataFrame для тестування
np.random.seed(42)
data = pd.DataFrame({
    'value': np.random.randint(1, 1000000, size=10**6)  # 1 млн випадкових чисел
})

# Функція для обробки числових значень
def process_value(x):
    return int(x ** 2) if x % 2 == 0 else int(x ** 3)  # Примусово перетворюємо в int

# Використання .apply()
start_time = time.time()
data['processed_apply'] = data['value'].apply(process_value)
apply_time = time.time() - start_time
print(f"Час виконання з .apply(): {apply_time:.2f} сек.")

# Оптимізація через numpy.vectorize()
vectorized_process = np.vectorize(process_value, otypes=[object])  # Використовуємо object для великих чисел

start_time = time.time()
data['processed_vectorized'] = vectorized_process(data['value'])
vectorize_time = time.time() - start_time
print(f"Час виконання з numpy.vectorize(): {vectorize_time:.2f} сек.")

# Порівняння швидкості
speedup = apply_time / vectorize_time
print(f"Прискорення завдяки vectorize: ~{speedup:.2f}x разів швидше.")
