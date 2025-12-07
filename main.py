import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model, to_categorical

# Завантаження та обробка даних
df = pd.read_csv("airports.csv")

cols_to_drop = ['gps_code', 'iata_code', 'local_code', 'home_link', 'wikipedia_link', 'keywords']
df = df.drop(columns=cols_to_drop)

df["continent"] = df["continent"].fillna("NA")
df = df.dropna()

# --- 1. ЗАДАЧА РЕГРЕСІЇ ---
print("--- РЕГРЕСІЯ ---")

X_reg = df[['latitude_deg', 'longitude_deg']].values
y_reg = df['elevation_ft'].values

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_reg = Sequential([
    Input(shape=(2,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model_reg.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_reg = model_reg.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Графіки для регресії
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_reg.history['loss'], label='Train Loss')
plt.plot(history_reg.history['val_loss'], label='Val Loss')
plt.title('Регресія: Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_reg.history['mae'], label='Train MAE')
plt.plot(history_reg.history['val_mae'], label='Val MAE')
plt.title('Регресія: MAE')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot передбачень
y_pred = model_reg.predict(X_test).flatten()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Реальна висота')
plt.ylabel('Передбачена')
plt.title('Реальність vs Прогноз')
plt.show()

# --- 2. ЗАДАЧА КЛАСИФІКАЦІЇ ---
print("\n--- КЛАСИФІКАЦІЯ ---")

X_cls = df[['latitude_deg', 'longitude_deg', 'elevation_ft']].values
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df['type'])
y_categorical = to_categorical(y_encoded)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_categorical, test_size=0.2, random_state=42)

scaler_c = StandardScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)

model_cls = Sequential([
    Input(shape=(3,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_cls = model_cls.fit(X_train_c, y_train_c, epochs=60, batch_size=32, validation_split=0.2, verbose=1)

# Графіки для класифікації
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_cls.history['loss'], label='Train Loss')
plt.plot(history_cls.history['val_loss'], label='Val Loss')
plt.title('Класифікація: Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_cls.history['accuracy'], label='Train Accuracy')
plt.plot(history_cls.history['val_accuracy'], label='Val Accuracy')
plt.title('Класифікація: Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Збереження структури моделей
# plot_model(model_reg, to_file='model_reg.png', show_shapes=True)
# plot_model(model_cls, to_file='model_cls.png', show_shapes=True)