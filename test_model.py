from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Path to your model
MODEL_PATH = "E:/COVID-19 DETECTION/Covid19-Detection-Seqential model/mico-html/New_Sequential_3.h5"

# Load the model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

def test_model(img_path):
    # Preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))  # Adjust size if necessary
    img_array = image.img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    preds = model.predict(img_array)
    print(f"Predictions: {preds}")

    # Interpret the result
    prediction = preds[0][0]  # Assuming binary classification
    if prediction > 0.5:
        result = f"Negative for COVID-19 (Confidence: {prediction:.2f})"
    else:
        result = f"Positive for COVID-19 (Confidence: {1 - prediction:.2f})"

    print(f"Result: {result}")

# Test with a known input image
test_img_path = "uploads/COVID-309.png"  # Replace with the path to your test image
test_model(test_img_path)
