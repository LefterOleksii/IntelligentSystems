import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

dataframe_path = "housing.csv"
dataframe = pd.read_csv(dataframe_path)

print(dataframe.info())

corr_matrix = dataframe.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm'
)
plt.title('Кореляційна матриця ознак датасету California Housing')
plt.show()

print("\nКореляція ознак з 'median_house_value':")
print(corr_matrix['median_house_value'].abs().sort_values(ascending=False))


initial_rows = dataframe.shape[0]
dataframe.dropna(inplace=True)
cleaned_rows = dataframe.shape[0]
print(f"Видалено {initial_rows - cleaned_rows} рядків з пропущеними значеннями.")


features_list = ['median_income', 'latitude']
target_variable = 'median_house_value'

X = dataframe[features_list]
y = dataframe[target_variable]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

print("\nПобудова візуалізації навчального датасету")

plt.figure(figsize=(10, 7))
sc = plt.scatter(
    X_train['median_income'],
    X_train['latitude'],
    c=y_train,
    alpha=0.4
)
plt.colorbar(sc, label='Median House Value ($)')
plt.xlabel('Median Income')
plt.ylabel('Latitude')
plt.title('Візуалізація навчального датасету (Ціна житла в залежності від доходу та широти)')
plt.grid(True)
plt.show()

print("\nНавчання моделі k-Найближчих Сусідів (KNN)")
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

print("\nОцінка моделі на тестових даних")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nПоказник якості: Середня Абсолютна Помилка (MAE)")
print(f"MAE: ${mae:,.2f}")


min_price = y.min()
max_price = y.max()

print("\nЗагальний діапазон цін в усьому датасеті (y)")
print(f"Мінімальна вартість: ${min_price:,.2f}")
print(f"Максимальна вартість: ${max_price:,.2f}")


