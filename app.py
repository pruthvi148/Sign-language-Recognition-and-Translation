from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import os
from io import BytesIO

app = Flask(__name__)
app.secret_key = "slr_secret_key"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("asl_alphabet_model.keras")

class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ---------------- LOGIN ---------------- #

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form["email"]
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- DASHBOARD ---------------- #

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# ---------------- IMAGE UPLOAD ---------------- #

@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    file = request.files["image"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = Image.open(filepath).resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = class_labels[np.argmax(prediction)]

    return {"prediction": result}

# ---------------- CAMERA PREDICTION ---------------- #

@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])

    img = Image.open(BytesIO(image_data)).convert("RGB")
    img = img.resize((64, 64))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = class_labels[np.argmax(prediction)]

    return {"prediction": result}


if __name__ == "__main__":
    app.run(debug=True)
