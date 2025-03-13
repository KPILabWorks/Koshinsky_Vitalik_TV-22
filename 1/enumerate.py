from typing import Iterable, Tuple, Any
import itertools
import timeit

def my_enumerate(iterable: Iterable, start: int = 1) -> Iterable[Tuple[int, Any]]:

    for index, item in zip(itertools.count(start), iterable):
        yield index, item

# Тестування роботи my_enumerate
data = ['apple', 'banana', 'cherry']
for index, value in my_enumerate(data):
    print(f"{index}: {value}")
print()

# Додаткові приклади використання
animals_list = ['cat', 'dog', 'parrot']
for index, animal in my_enumerate(animals_list, start=1):
    print(f"{index}: {animal}")
print()

text = "Hello"
for index, char in my_enumerate(text):
    print(f"Character '{char}' is at index {index}")
print()

# Використання timeit для вимірювання продуктивності
execution_time = timeit.timeit(lambda: list(my_enumerate(data)), number=100000)
print(f"Час виконання my_enumerate: {execution_time:.5f} секунд")
