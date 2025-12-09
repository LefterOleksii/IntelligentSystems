import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)


MODEL_PATH = 'cifar_subset_model.h5'
model = load_model(MODEL_PATH)

# Класи, на яких навчали модель (порядок важливий!)
CLASS_NAMES = ['Airplane', 'Bird', 'Deer']


def prepare_image(img_bytes):
    """
    Функція підготовки зображення:
    Змінює розмір та нормалізує колір пікселів
    """
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


@app.route('/')
def home():
    """
    Відкриває html сторінку інтерфейсу
    """
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        processed_img = prepare_image(img_bytes)

        prediction = model.predict(processed_img)

        results = {}
        for i, class_name in enumerate(CLASS_NAMES):
            percent = float(prediction[0][i]) * 100
            results[class_name] = f"{percent:.2f}%"

        best_class_index = np.argmax(prediction)
        best_class = CLASS_NAMES[best_class_index]

        return jsonify({
            'prediction': best_class,
            'confidence': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
