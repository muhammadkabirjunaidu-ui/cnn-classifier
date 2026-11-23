import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

# Load your trained model once when the server starts
model = tf.keras.models.load_model('flashdrive_mouse_cnn.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # 1. Open image, convert to RGB, and resize to 64x64
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((64, 64))

        # 2. Convert to array, rescale (1./255), and expand batch dimension
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Run prediction (Alphabetical: 0 = flashdrive, 1 = mouse)
        prob = float(model.predict(img_array)[0][0])
        
        if prob >= 0.5:
            label = 'Mouse'
            confidence = prob * 100
        else:
            label = 'Flashdrive'
            confidence = (1 - prob) * 100

        return jsonify({
            'prediction': label,
            'confidence': f'{confidence:.1f}%'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
