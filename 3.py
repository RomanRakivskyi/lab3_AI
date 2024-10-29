import csv
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Завантаження даних із CSV-файлу
def read_csv_data(file_path):
    entries = []
    with open(file_path, newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Пропускаємо заголовок
        for line in csv_reader:
            # Конвертуємо числові значення у float, а клас залишаємо текстовим
            entries.append([float(line[j]) for j in range(4)] + [line[4]])
    return entries

# Випадкове перемішування даних
def randomize_data(entries):
    random.shuffle(entries)

# Нормалізація даних
def scale_features(entries):
    min_max_pairs = []
    for j in range(len(entries[0]) - 1):  # Виключаємо клас
        values = [line[j] for line in entries]
        min_val, max_val = min(values), max(values)
        min_max_pairs.append((min_val, max_val))
    for line in entries:
        for j in range(len(line) - 1):
            min_val, max_val = min_max_pairs[j]
            line[j] = (line[j] - min_val) / (max_val - min_val)

# Розрахунок Евклідової відстані
def calculate_distance(vec1, vec2):
    return math.sqrt(sum((vec1[i] - vec2[i]) ** 2 for i in range(len(vec1) - 1)))

# Знаходження k найближчих сусідів
def k_nearest_neighbors(train_data, test_point, k):
    distances = [(train_row, calculate_distance(test_point, train_row)) for train_row in train_data]
    distances.sort(key=lambda x: x[1])
    return [distances[i][0] for i in range(k)]

# Класифікація на основі сусідів
def classify_instance(train_data, test_point, k):
    nearest = k_nearest_neighbors(train_data, test_point, k)
    labels = [neighbor[-1] for neighbor in nearest]
    return max(set(labels), key=labels.count)

# Обчислення точності
def evaluate_accuracy(test_data, predicted_labels):
    correct_count = sum(1 for i in range(len(test_data)) if test_data[i][-1] == predicted_labels[i])
    return (correct_count / len(test_data)) * 100

# Розділення на навчальну та тестову вибірки
def split_data(data, ratio=0.7):
    split_idx = int(len(data) * ratio)
    training_data = random.sample(data, split_idx)
    test_data = [entry for entry in data if entry not in training_data]
    return training_data, test_data

# Головна частина програми
file_path = 'IrisData_full.csv'
data = read_csv_data(file_path)

# Підготовка даних
randomize_data(data)
scale_features(data)

# Розділення на навчальну та тестову вибірки
train_data, test_data = split_data(data, ratio=0.7)

# Оцінка точності для різних значень k
k_results = []
for k in range(1, 11):
    test_predictions = [classify_instance(train_data, test_instance, k) for test_instance in test_data]
    acc = evaluate_accuracy(test_data, test_predictions)
    k_results.append((k, acc))
    print(f'Точність при K={k}: {acc:.2f}%')

# Візуалізація парних графіків із індивідуальними параметрами
df = pd.DataFrame(data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
sns.pairplot(df, hue="class", palette="husl", markers=["o", "s", "D"])
plt.suptitle("Парні графіки характеристик Ірисів", y=1.02)
plt.show()

# Графік точності для різних значень k з налаштуванням стилю
k_values, accuracy_scores = zip(*k_results)
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', color='purple', linewidth=2, markersize=6)
plt.xlabel("Значення K")
plt.ylabel("Точність (%)")
plt.title("Точність K-NN при різних значеннях K")
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()
