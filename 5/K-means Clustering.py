import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.animation as animation
import pandas as pd

# Зчитуємо дані з CSV-файлу
df = pd.read_csv("cluster_data.csv")
X = df.values  # Перетворюємо у формат numpy

# Візуалізуємо початкові дані
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], s=50)
plt.title("Початкові дані")
plt.show()

# Кількість кластерів
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Створюємо кольорову палітру для кластерів
palette = sns.color_palette("husl", k)
colors = [palette[label] for label in labels]

# Візуалізуємо кластери
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette=palette, s=50, alpha=0.7, edgecolor='black', legend=False)
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='red', marker='X', s=200, label='Центроїди')
plt.title("Результати кластеризації методом K-means")
plt.legend()
plt.show()

# Додаємо анімацію процесу кластеризації
def update(frame):
    global kmeans
    kmeans = KMeans(n_clusters=k, init='random', n_init=1, max_iter=frame + 1, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    ax.clear()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette=palette, s=50, alpha=0.7, edgecolor='black', legend=False, ax=ax)
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='red', marker='X', s=200, label='Центроїди', ax=ax)
    ax.set_title(f"Ітерація {frame + 1}")
    ax.legend()

fig, ax = plt.subplots(figsize=(8, 6))
ani = animation.FuncAnimation(fig, update, frames=10, interval=700, repeat=False)
plt.show()
