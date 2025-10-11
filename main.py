import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from functions import *

dataframe_path = "airports.csv"
airports_dataframe = pd.read_csv(dataframe_path)

# print(airports_dataframe.isnull().sum())
# print(airports_dataframe.duplicated().sum())
# print(airports_dataframe.info())
# print(airports_dataframe.describe())

cols_to_drop = ['gps_code', 'iata_code', 'local_code', 'home_link', 'wikipedia_link', 'keywords']
airports_dataframe = airports_dataframe.drop(columns=cols_to_drop)

# print(airports_dataframe.info())
# print(airports_dataframe.describe())

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

    # airports_dataframe = fill_missing_with_mean_or_mode(airports_dataframe)
    # airports_dataframe = drop_rows_with_missing(airports_dataframe)



    # get_histogram_for_numeric()
    # get_barplot_for_objects()
    get_boxplots_for_numeric()


    # get_correlation_matrix_of_dataset_attributes()
    # get_space_of_informative_signs()
    # get_3d_space_of_informative_signs()
    # get_informative_bloxplots()

    # Crosstab + stacked bar
    # Таблиця спряженості


