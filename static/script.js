const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');

imageInput.onchange = evt => {
    const [file] = imageInput.files;
    if (file) {
        imagePreview.src = URL.createObjectURL(file);
        imagePreview.style.display = 'inline-block';
        resultDiv.style.display = 'none';

        document.querySelector('.custom-file-upload').innerText = "✅ " + file.name;
    }
};

form.onsubmit = async (e) => {
    e.preventDefault();
    const file = imageInput.files[0];
    if (!file) {
        alert("Будь ласка, оберіть файл!");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    resultDiv.style.display = 'block';
    resultDiv.innerHTML = 'Обробка...';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<span style="color:red">Помилка: ${data.error}</span>`;
        } else {
            let html = `<h3>Результат: ${data.prediction}</h3><ul>`;
            for (const [cls, score] of Object.entries(data.confidence)) {
                const style = cls === data.prediction ? 'font-weight:bold; color:green;' : '';
                html += `<li style="${style}">${cls}: ${score}</li>`;
            }
            html += '</ul>';
            resultDiv.innerHTML = html;
        }
    } catch (err) {
        resultDiv.innerHTML = `<span style="color:red">Помилка з'єднання</span>`;
    }
};