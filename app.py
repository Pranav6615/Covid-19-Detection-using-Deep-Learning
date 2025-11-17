##NEW CODE :: 

from __future__ import division, print_function
import sys
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Define a Flask app
app = Flask(__name__)

# Get the current folder where app.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Join it with the model name
model_path = os.path.join(base_dir, 'New_Sequential_3.keras')

model = load_model(model_path, compile=False)
model.make_predict_function()  # Necessary for Keras

print('Model loaded. Start serving...')

def model_predict(img_path, model):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img,(150,150))
        img_array = np.array(img)
        img_array.shape

        img_array = img_array.reshape(1,150,150,3)
        img_array.shape
        # Make predictions
        a=model.predict(img_array)
        indices = a.argmax()
        indices
        if indices==0:
            return "COVID"
        else:
            return "Normal"
    except Exception as e:
        print(f"Error in model prediction: {e}")
        raise

# Route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Additional routes for static pages
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/doctor', methods=['GET'])
def doctor():
    return render_template('doctor.html')

@app.route('/testimonial', methods=['GET'])
def testimonial():
    return render_template('testimonial.html')

@app.route('/treatment', methods=['GET'])
def treatment():
    return render_template('treatment.html')

# Route for the prediction page
@app.route('/predict', methods=['POST'])
def upload():
    try:
        # Check if 'file' exists in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        f = request.files['file']

        # Check if the file is selected
        if f.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)  # Create uploads folder if it doesn't exist

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)

        print(result)
        return jsonify({'prediction':result})
    

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



