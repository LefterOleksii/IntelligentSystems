import pandas
from matplotlib import pyplot as plt

dataframe_path = "train_and_test2.csv"

titanic_dataframe = pandas.read_csv(dataframe_path)

print(100*"/")
print("Виконання функції read_csv")
print(100*"-")
print(titanic_dataframe)

# ------------------------------
# Ознайомлення зі структурою даних датасета за допомогою функцій head, info, describe.
# ------------------------------
print(100*"/")
print("Ознайомлення зі структурою датасету за допомогою функції head")
print(100*"-")
print(titanic_dataframe.head())

print(100*"/")
print("Ознайомлення зі структурою датасету за допомогою функції info")
print(100*"-")
print(titanic_dataframe.info())

print(100*"/")
print("Ознайомлення зі структурою датасету за допомогою функції describe")
print(100*"-")
print(titanic_dataframe.describe())

# ------------------------------
# Ознайомлення з розмірністю та типами даних датасета за допомогою атрибутів shape, dtypes.
# ------------------------------
print(100*"/")
print("Ознайомлення із розмірністю та типами даних датасету за допомогою атрибуту shape")
print(100*"-")
print(titanic_dataframe.shape)

print(100*"/")
print("Ознайомлення із розмірністю та типами даних датасету за допомогою атрибуту dtypes")
print(100*"-")
print(titanic_dataframe.dtypes)
print(100*"/")


# ------------------------------
# Додавання до датафрейму поля Age_group як поле Age з градацією у 10 років.
# ------------------------------
# Створюємо межі вікових груп (0-9, 10-19, ..., 90-99, 100)
age_groups = range(0, 101, 10)  # від 0 до 100 з кроком 10
# Створюємо мітки для кожної вікової групи
labels = [f"{i}-{i+9}" for i in age_groups[:-1]]
# Створюємо нове поле 'Age_group', де віки розбиті по групах
titanic_dataframe['Age_group'] = pandas.cut(titanic_dataframe['Age'], bins=age_groups, labels=labels)
# Перевіряємо результат
print(titanic_dataframe[['Age', 'Age_group']].head())

# ------------------------------
# Розрахунок кількості пасажирів за кожною групою по полям 'Sex', 'Age_group', 'Survived' за допомогою функції groupby.
# ------------------------------
grouped = titanic_dataframe.groupby(['Sex', 'Age_group', '2urvived'], observed=True).size().reset_index(name='Passenger_count')
print(100*"/")
print("Групування пасажирів по статі, віку та виживанню:")
print(grouped)
print(100*"/")

# ------------------------------
# Візуалізація кількості виживших та не виживших чоловіків і жінок по віковій групі
# ------------------------------
grouped['Sex_Survived'] = grouped['Sex'].map({0: 'Male', 1: 'Female'}) + '_' + grouped['2urvived'].map(
    {0: 'Died', 1: 'Survived'})

pivot_table = grouped.pivot_table(
    index='Age_group',
    columns='Sex_Survived',
    values='Passenger_count',
    fill_value=0,
    observed=True
)

column_order = ['Male_Died', 'Male_Survived', 'Female_Died', 'Female_Survived']
# Додаємо відсутні колонки з нулями
for col in column_order:
    if col not in pivot_table.columns:
        pivot_table[col] = 0

pivot_table = pivot_table[column_order]

pivot_table.plot(
    kind='bar',
    title='Кількість пасажирів за віковими групами, статтю та виживанням',
    figsize=(14, 8),
    color=['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
)

plt.xlabel("Вікові групи")
plt.ylabel("Кількість пасажирів")
plt.xticks(rotation=45)
plt.legend(title='Стать та виживання', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# plt.show()



# ------------------------------
# Визначаємо кількість врятованих та відсоток відношення врятованих до загиблих для кожної групи пасажирів.
# Визначаємо групу пасажирів з максимальною кількістю врятованих та максимальним відсотком відношення врятованих до загиблих.
# ------------------------------
pivot = grouped.pivot_table(index=['Sex','Age_group'], columns='2urvived', values='Passenger_count', fill_value=0, observed=True)

# Перейменуємо колонки: 0 = Dead, 1 = Survived
pivot = pivot.rename(columns={0:'Dead', 1:'Survived'})

# Додаємо колонку з відсотком виживших від загиблих
pivot['Survived_to_Dead_%'] = (pivot['Survived'] / pivot['Dead'] * 100).replace([float('inf')], 0)

print("Аналіз виживання по групах (кількість та %):")
print(pivot)

# Знаходимо групу з максимальною кількістю виживших
max_survived_group = pivot['Survived'].idxmax()
max_survived_count = pivot['Survived'].max()

# Знаходимо групу з максимальним відсотком виживших до загиблих
max_ratio_group = pivot['Survived_to_Dead_%'].idxmax()
max_ratio_value = pivot['Survived_to_Dead_%'].max()

print("\nГрупа з максимальною кількістю врятованих: {} вікова група {} -> {} осіб".format(
    'Male' if max_survived_group[0] == 0 else 'Female', max_survived_group[1], max_survived_count))

print("Група з максимальним відсотком врятованих до загиблих: {} вікова група {} -> {:.2f}%".format(
    'Male' if max_ratio_group[0] == 0 else 'Female', max_ratio_group[1], max_ratio_value))

