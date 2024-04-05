import sys
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from tensorflow import keras

if sys.stdout.encoding != 'UTF-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='UTF-8', buffering=1)

app = Flask(__name__)

try:
    model = keras.models.load_model('model_final.h5')
    class_labels = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Phytophthora', 'Pest', 'Virus']
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        image = Image.open(file)
        image = image.resize((224, 224))  # Resize image to match model input size
        image = np.array(image) / 255.0    # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction using your model
        prediction = model.predict(image)

        # Get the predicted class and confidence
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])

        # Prepare response
        response = {'predicted_class': predicted_class, 'confidence': confidence}

        # Return the prediction as JSON
        return jsonify(response)

    except OSError as e:
        print("OS Error during prediction:", e)
        return jsonify({'error': 'An OS error occurred during prediction'})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'An error occurred during prediction'})

if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0', threaded=False)
