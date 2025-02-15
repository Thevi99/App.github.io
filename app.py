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
CORS(app, resources={r"/predict": {"origins": "https://thevi99.github.io"}})

# URL ไฟล์โมเดลบน GitHub LFS (แก้ไขให้ตรงกับ Repo ของคุณ)
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
print("🔍 กำลังโหลดโมเดลเข้าสู่ระบบ...")
model = tf.keras.models.load_model(model_path)
print("✅ โมเดลพร้อมใช้งาน!")

# หมวดหมู่
categories = ["Class A", "Class B", "Class C", "Notsandimage"]

def preprocess_image(image_file):
    """แปลงไฟล์ภาพให้เป็น Input ที่สามารถใช้กับโมเดลได้"""
    img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    img = img.resize((765, 1020))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

@app.route("/")
def index():
    """เสิร์ฟไฟล์ HTML"""
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    """API สำหรับพยากรณ์รูปภาพ"""
    print("📥 รับคำขอใหม่ที่ /predict")

    # เช็คว่ามีไฟล์ใน request หรือไม่
    if "image" not in request.files:
        print("🚨 ไม่มีไฟล์ที่ชื่อ 'image' ถูกส่งมา!")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["image"]
    print(f"🖼️ ไฟล์ที่ได้รับ: {file.filename}")

    if file.filename == "":
        print("🚨 ไม่มีไฟล์ถูกเลือก!")
        return jsonify({"error": "No selected file"}), 400
    
    try:
        print("🔄 กำลังประมวลผลภาพ...")
        img_array, original_img = preprocess_image(file)

        print("🤖 กำลังพยากรณ์ผลลัพธ์จากโมเดล...")
        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]
        print(f"✅ ผลลัพธ์การพยากรณ์: {predicted_class}")

        # แปลงภาพเป็น Base64
        buffered = io.BytesIO()
        original_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "class": predicted_class,
            "prediction": prediction.tolist(),
            "image_data": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        return jsonify({"error": str(e)}), 500

# รันเซิร์ฟเวอร์บน Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 กำลังรัน Flask บนพอร์ต {port}")
    app.run(host="0.0.0.0", port=port)
