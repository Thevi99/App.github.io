import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

model = tf.keras.models.load_model("sand_classification_model20-10.h5")

categories = ["Class A", "Class B", "Class C", "Notsandimage"]

def preprocess_image(image_file):
    img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    img = img.resize((765, 1020))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    try:
        # ประมวลผลภาพ
        img_array, original_img = preprocess_image(file)
        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]
        
        # แปลงภาพเป็น base64
        buffered = io.BytesIO()
        original_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            "class": predicted_class,
            "image_data": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
