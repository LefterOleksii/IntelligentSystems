import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import optimizers
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.metrics import classification_report, confusion_matrix


(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
target_classes = [0, 2, 4]
class_names_str = ['Airplane', 'Bird', 'Deer']


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

num_classes = len(target_classes)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Тренувальна вибірка: {x_train.shape}")
print(f"Тестова вибірка: {x_test.shape}")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

try:
    plot_model(model, to_file='model_structure.png', show_shapes=True)
except:
    print("Помилка генерації структури моделі")


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=100,
                    validation_data=(x_test, y_test),
                    verbose=1)

plt.figure(figsize=(12, 4))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Навчання (Train)')
plt.plot(history.history['val_accuracy'], label='Тест (Val)')
plt.title('Точність моделі')
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
plt.show()


loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nФінальна точність: {accuracy * 100:.2f}%")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

try:
    print(classification_report(y_true, y_pred_classes, target_names=class_names_str))
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_str, yticklabels=class_names_str)
    plt.title('Матриця плутанини')
    plt.ylabel('Справжній клас')
    plt.xlabel('Передбачений клас')
    plt.show()
except NameError:
    print("Помилка")