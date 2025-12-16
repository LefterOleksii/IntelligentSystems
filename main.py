import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import optimizers
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

(x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()

target_classes = [7, 9]
class_names_str = ['Sneaker', 'Ankle boot']


def filter_data(x, y, classes):
    mask = np.isin(y, classes).flatten()
    x_filtered = x[mask]
    y_filtered = y[mask]

    y_new = np.zeros_like(y_filtered)
    for i, original_label in enumerate(classes):
        y_new[y_filtered == original_label] = i
    return x_filtered, y_new


x_train, y_train = filter_data(x_train_full, y_train_full, target_classes)
x_test, y_test = filter_data(x_test_full, y_test_full, target_classes)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

num_classes = len(target_classes)  # = 2
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"Тренувальна вибірка: {x_train.shape}")
print(f"Тестова вибірка: {x_test.shape}")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'mse'])

model.summary()

history = model.fit(x_train, y_train_cat,
                    epochs=15,
                    batch_size=64,
                    validation_data=(x_test, y_test_cat),
                    verbose=1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Навчання (Train)')
plt.plot(history.history['val_accuracy'], label='Тест (Val)')
plt.title('Точність (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Навчання (Train)')
plt.plot(history.history['val_loss'], label='Тест (Val)')
plt.title('Втрати (Loss)')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Оцінка моделі ---")
loss, accuracy, mse_score = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

calc_mse = mean_squared_error(y_test_cat, y_pred_probs)
print(f"MSE (Середньоквадратична помилка): {calc_mse:.5f}")

calc_r2 = r2_score(y_test_cat, y_pred_probs)
print(f"R² Score: {calc_r2:.5f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names_str))

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names_str, yticklabels=class_names_str)
plt.title('Матриця плутанини (Confusion Matrix)')
plt.ylabel('Справжній клас')
plt.xlabel('Передбачений клас')
plt.show()