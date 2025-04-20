from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
from utils import preprocess_genre_input, preprocess_title_overview_input, preprocess_poster_input
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Dummy model — replace with yours
class RevenueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = RevenueModel()
model.eval()

GENRE_CHOICES = [
    "Action", "Comedy", "Drama", "Fantasy", "Horror",
    "Romance", "Sci-Fi", "Thriller", "Animation", "Documentary"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        genre = request.form["genre"]
        title = request.form["title"]
        overview = request.form["overview"]
        poster_file = request.files["poster"]

        poster_path = None
        if poster_file:
            filename = secure_filename(poster_file.filename)
            poster_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            poster_file.save(poster_path)

        # Preprocess all inputs into a model-ready tensor
        genre_tensor = preprocess_genre_input(genre)
        title_overview_tensor = preprocess_title_overview_input(title, overview)
        poster_tensor = preprocess_poster_input(poster_path)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = float(output.item())

    return render_template("index.html", genres=GENRE_CHOICES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
