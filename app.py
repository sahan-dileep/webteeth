from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO("best.pt")  # Replace with the path to your trained model

# Define the upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if an image was uploaded
        if "image" not in request.files:
            return "No image uploaded", 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return "No image selected", 400

        # Save the uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)

        # Run YOLO inference
        results = model(image_path)

        # Save the results
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
        results[0].save(filename=output_path)

        return render_template("result.html", result_image=output_path)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)