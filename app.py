from flask import Flask, render_template, request
import os
import pandas as pd
from ultralytics import YOLO

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO Classification Model
# It will download automatically the first time
model = YOLO("yolov8n-cls.pt")

# Read animal information
animal_data = pd.read_csv("animal_info.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(image_path)

    # Predict
    results = model(image_path)

    probs = results[0].probs

    class_id = probs.top1
    confidence = float(probs.top1conf) * 100

    species = results[0].names[class_id].lower()

    # Search animal info
    info = animal_data[animal_data["Species"].str.lower() == species]

    if not info.empty:
        animal = info.iloc[0]

        return render_template(
            "result.html",
            image=image_path,
            species=species.title(),
            confidence=round(confidence, 2),
            scientific_name=animal["Scientific_Name"],
            diet=animal["Diet"],
            habitat=animal["Habitat"],
            lifespan=animal["Lifespan"],
            status=animal["Status"],
            fact=animal["Fact"],
        )

    else:
        return render_template(
            "result.html",
            image=image_path,
            species=species.title(),
            confidence=round(confidence, 2),
            scientific_name="Not Available",
            diet="Not Available",
            habitat="Not Available",
            lifespan="Not Available",
            status="Not Available",
            fact="No information found.",
        )


if __name__ == "__main__":
    app.run(debug=True)
