from matplotlib import pyplot as plt

from main import airports_dataframe, numeric_cols, labels_ukr, categorical_cols
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px

# Виводить гістограми та щільність для числових ознак датафрейму
def get_histogram_for_numeric():
    for col in numeric_cols:
        plt.figure(figsize=(15,4), dpi=200)
        sns.histplot(airports_dataframe[col], kde=True)
        plt.title(f'Гістограма та щільність розподілу для {labels_ukr[col]}')
        plt.xlabel(labels_ukr[col])
        plt.ylabel('Кількість')
        plt.show()


# Виводить стовпчикові графіки для категорійних ознак датасету
def get_barplot_for_objects():
    for col in categorical_cols:
        plt.figure(figsize=(20, 6), dpi=200)
        counts = airports_dataframe[col].value_counts()
        if len(counts) > 20:
            top_counts = counts[:20]
            others_count = counts[20:].sum()
            others_num_categories = len(counts[20:])
            top_counts[f'Інше ({others_num_categories})'] = others_count
        else:
            top_counts = counts

        sns.barplot(x=top_counts.index, y=top_counts.values)

        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'Розподіл {labels_ukr[col]}' + (' (топ 20 + Інше)' if len(counts) > 20 else ''), fontsize=14)
        plt.ylabel('Кількість аеропортів', fontsize=12)
        plt.xlabel(labels_ukr[col], fontsize=12)
        plt.tight_layout()
        plt.show()


# Виводить віскерні графіки для числових ознак
def get_boxplots_for_numeric():
    fig = plt.figure(constrained_layout=True, figsize=(20,10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :])
    ]
    for col, ax in zip(numeric_cols, axes):
        sns.boxplot(x=airports_dataframe[col], ax=ax)
        ax.set_title(f'Віскерний графік для {labels_ukr[col]}')
    plt.show()


# Виводить кореляційну матрицю полів датафрейму
def get_correlation_matrix_of_dataset_attributes():
    df_encoded = airports_dataframe[numeric_cols + categorical_cols].copy()

    for col in categorical_cols:
        freq_map = df_encoded[col].value_counts().to_dict()
        df_encoded[col] = df_encoded[col].map(freq_map)

    corr_full = df_encoded.corr()

    plt.figure(figsize=(12,8), dpi=150)
    sns.heatmap(corr_full, annot=True, cmap='coolwarm', fmt=".2f")

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.title('Кореляційна матриця')
    plt.show()


# Виводить простір інформаційних ознак
def get_space_of_informative_signs():
    informative_cols = ['latitude_deg', 'longitude_deg']
    target = 'type'

    plt.figure(figsize=(10,6), dpi=150)
    sns.scatterplot(
        data=airports_dataframe,
        x=informative_cols[1],
        y=informative_cols[0],
        hue=airports_dataframe[target],
        palette='Set2'
    )
    plt.title('Відображення класів у просторі інформативних ознак')
    plt.xlabel('Широта')
    plt.ylabel('Довгота')
    plt.legend(title='Тип аеропорту')
    plt.show()


# Виводить 3D простір інформаційних ознак
def get_3d_space_of_informative_signs():
    top_types = airports_dataframe['type'].value_counts().head(5).index
    df_filtered = airports_dataframe[airports_dataframe['type'].isin(top_types)].copy()

    fig = px.scatter_3d(
        df_filtered,
        x='latitude_deg',
        y='longitude_deg',
        z='elevation_ft',
        color='type',
        title='3D відображення типів аеропортів',
        labels={
            'latitude_deg': 'Широта',
            'longitude_deg': 'Довгота',
            'elevation_ft': 'Висота (ft)'
        },
        opacity=0.6
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        scene=dict(
            xaxis_title='Широта',
            yaxis_title='Довгота',
            zaxis_title='Висота (ft)',
            aspectratio=dict(x=2, y=2, z=1.5)
        ),
        legend=dict(
            title=dict(text='Тип аеропорту', font=dict(size=16)),
            font=dict(size=14),
            itemsizing='constant'
        )
    )
    fig.update_scenes(xaxis_autorange="reversed")
    fig.show()


# Виводить інші віскерні графіки
def get_informative_bloxplots():
    informative_cols = ['elevation_ft', 'latitude_deg', 'longitude_deg']
    target = 'type'

    for col in informative_cols:
        plt.figure(figsize=(20,13), dpi=150)
        sns.boxplot(x=target, y=col, data=airports_dataframe)
        plt.title(f'Віскерний графік {col} по типу аеропорту')
        plt.xticks(rotation=45)
        plt.show()


# Замінює порожні значення на середні по числовим, та випадкові з категорійних
def fill_missing_with_mean_or_mode(df: pd.DataFrame) -> pd.DataFrame:
    df_filled = df.copy()
    for col in df.columns:
        missing_count = df_filled[col].isna().sum()
        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                mean_value = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(mean_value)
                print(f"[+] Колонка '{col}' мала {missing_count} пропусків → заповнено середнім ({mean_value:.2f})")
            else:
                # Беремо розподіл існуючих значень
                value_counts = df_filled[col].value_counts(normalize=True, dropna=True)

                if not value_counts.empty:
                    # Генеруємо рандомні значення з урахуванням ймовірностей
                    random_values = np.random.choice(
                        value_counts.index,
                        size=missing_count,
                        p=value_counts.values
                    )

                    # Заповнюємо пропуски
                    mask = df_filled[col].isna()
                    df_filled.loc[mask, col] = random_values
                    print(f"[+] Колонка '{col}' мала {missing_count} пропусків → заповнено рандомно за розподілом")
                else:
                    df_filled[col] = df_filled[col].fillna('Unknown')
                    print(f"[+] Колонка '{col}' мала {missing_count} пропусків → заповнено 'Unknown'")
        else:
            print(f"[=] Колонка '{col}' повністю заповнена")
    return df_filled


# Видаляє рядки у яких є порожні клітинки
def drop_rows_with_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df_clean = df.dropna()
    after = len(df_clean)
    return df_clean