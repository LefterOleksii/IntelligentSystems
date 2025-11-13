import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, \
    recall_score, f1_score
from sklearn.metrics import confusion_matrix


from functions import *

dataframe_path = "airports.csv"
airports_dataframe = pd.read_csv(dataframe_path)

cols_to_drop = ['gps_code', 'iata_code', 'local_code', 'home_link', 'wikipedia_link', 'keywords']
airports_dataframe = airports_dataframe.drop(columns=cols_to_drop)
numeric_cols = ['latitude_deg', 'longitude_deg', 'elevation_ft']
categorical_cols = ['type', 'iso_country', 'scheduled_service', 'municipality', 'iso_region', "continent"]

labels_ukr = {
    'latitude_deg': 'Широта (°)',
    'longitude_deg': 'Довгота (°)',
    'elevation_ft': 'Висота (фт)',
    'type': 'Тип аеропорту',
    'continent': 'Континент',
    'iso_country': 'Код країни',
    'iso_region': 'Код регіону',
    'municipality': 'Муніципалітет',
    'scheduled_service': 'Регулярне обслуговування'
}







if __name__ == "__main__":
    # Виправляє значення датафрейму
    airports_dataframe["continent"] = airports_dataframe["continent"].fillna("NA")
    airports_dataframe = drop_rows_with_missing(airports_dataframe)

    # Побудова моделі
    top_types = airports_dataframe['type'].value_counts().index
    df_filtered = airports_dataframe[airports_dataframe['type'].isin(top_types)].copy()
    X = df_filtered[['latitude_deg', 'longitude_deg', 'elevation_ft']]
    y = df_filtered['type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    # Візуалізація дерева рішень
    plt.figure(figsize=(35, 12))
    plot_tree(model,
              feature_names=['latitude_deg', 'longitude_deg', 'elevation_ft'],
              class_names=model.classes_,
              filled=True,
              fontsize=10,
              rounded=True,
              proportion=True)
    plt.title("Дерево рішень для класифікації", fontsize=18)
    plt.tight_layout()
    plt.show()

    # Показники якості дерева рішень
    print("\n" + "=" * 80)
    print("ПОКАЗНИКИ ЯКОСТІ ДЕРЕВА РІШЕНЬ")
    print("=" * 80)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nЗвіт класифікації:")
    print(classification_report(y_test, y_pred, target_names=model.classes_, zero_division=0))

    print("\nВажливість ознак:")
    for feature, importance in zip(['latitude_deg', 'longitude_deg', 'elevation_ft'],
                                   model.feature_importances_):
        print(f"  {feature:15s}: {importance:.4f} ({importance * 100:.1f}%)")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title('Матриця помилок дерева рішень', fontsize=14)
    plt.xlabel('Передбачений клас')
    plt.ylabel('Справжній клас')
    plt.tight_layout()
    plt.show()

    print("=" * 80)


    # Побудова моделі KNN
    print("\n\n" + "=" * 80)
    print("ПОБУДОВА МОДЕЛІ KNN")
    print("=" * 80)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    print(f"\nМодель навчена (n_neighbors=5)")
    print(f"Тренувальна вибірка: {len(X_train)}")
    print(f"Тестова вибірка: {len(X_test)}")
    print("=" * 80)

    # Візуалізація KNN
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
                xticklabels=knn_model.classes_,
                yticklabels=knn_model.classes_)
    plt.title('Матриця помилок KNN', fontsize=14)
    plt.xlabel('Передбачений клас')
    plt.ylabel('Справжній клас')
    plt.tight_layout()
    plt.show()

    # Показники якості KNN
    print("\n" + "=" * 80)
    print("ПОКАЗНИКИ ЯКОСТІ МОДЕЛІ KNN")
    print("=" * 80)

    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"\nAccuracy: {accuracy_knn:.4f} ({accuracy_knn * 100:.2f}%)")

    print("\nЗвіт класифікації:")
    print(classification_report(y_test, y_pred_knn, target_names=knn_model.classes_, zero_division=0))

    precision_knn = precision_score(y_test, y_pred_knn, average='weighted', zero_division=0)
    recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
    f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

    print("Зведені метрики:")
    print(f"  Precision: {precision_knn:.4f}")
    print(f"  Recall:    {recall_knn:.4f}")
    print(f"  F1-Score:  {f1_knn:.4f}")

    print("=" * 80)


    # Порівняння моделей
    print("\n\n" + "=" * 80)
    print("ПОРІВНЯННЯ МОДЕЛЕЙ КЛАСИФІКАЦІЇ")
    print("=" * 80)

    print(f"\nDecision Tree:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print(f"\nKNN:")
    print(f"  Accuracy:  {accuracy_knn:.4f} ({accuracy_knn * 100:.2f}%)")
    print(f"  Precision: {precision_knn:.4f}")
    print(f"  Recall:    {recall_knn:.4f}")
    print(f"  F1-Score:  {f1_knn:.4f}")

    print(f"\nРізниця Accuracy: {abs(accuracy - accuracy_knn):.4f} ({abs(accuracy - accuracy_knn) * 100:.2f}%)")


    # Візуалізація порівняння
    models = ['Decision Tree', 'KNN']
    accuracies = [accuracy, accuracy_knn]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['steelblue', 'seagreen'], alpha=0.8)
    plt.ylim([0, 1])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Порівняння точності моделей', fontsize=14)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{acc:.4f}\n({acc * 100:.2f}%)',
                 ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.show()

    print("=" * 80)





