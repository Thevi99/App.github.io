import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# URL ไฟล์โมเดลบน GitHub LFS (แก้ไข URL ให้ตรงกับ Repo ของคุณ)
model_url = "https://github.com/Thevi99/App.github.io/raw/main/sand_classification_model20-10.h5"
model_path = "sand_classification_model20-10.h5"

# โหลดโมเดลจาก GitHub LFS ถ้ายังไม่มี
if not os.path.exists(model_path):
    print("🚀 กำลังโหลดโมเดลจาก GitHub LFS...")
    response = requests.get(model_url, stream=True)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ โหลดโมเดลสำเร็จ!")
    else:
        raise Exception(f"❌ โหลดโมเดลล้มเหลว! Status code: {response.status_code}")

# โหลดโมเดล
model = tf.keras.models.load_model(model_path)

# หมวดหมู่
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
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    try:
        img_array, original_img = preprocess_image(file)
        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]

        buffered = io.BytesIO()
        original_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "class": predicted_class,
            "prediction": prediction.tolist(),
            "image_data": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# รันเซิร์ฟเวอร์บน Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
