from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
from utils import preprocess_genre_input, preprocess_title_overview_input, preprocess_poster_input
from werkzeug.utils import secure_filename
import os
import pickle

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# GENRE --- GENERIC FEATURES MODEL
generic_model = pickle.load(open("models/generic_features_model.pkl", "rb"))

# TITLE/OVERVIEW --- TWO TOWER BERT MODEL

# POSTER --- VISUAL ENSEMBLE MODEL

# LATE FUSION --- GATING ENSEMBLE MODEL



GENRE_CHOICES = ['Drama', 'Romance', 'Horror', 'Thriller', 'Action', 'Adventure',
 'Science Fiction', 'Crime', 'Comedy', 'History', 'War',
 'Mystery', 'Fantasy', 'Family', 'Animation', 'Western', 'Music',
 'Documentary', 'TV Movie']
MONTH_CHOICES = [1,2,3,4,5,6,7,8,9,10,11,12]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = {
            "title": request.form["title"],
            "overview": request.form["overview"],
            "budget": float(request.form["budget"]),
            "genre": request.form.getlist("genre"),
            "release_month": int(request.form["release_month"]),
            "runtime": int(request.form["runtime"]),
            "view_count": int(request.form["view_count"]),
            "like_count": int(request.form["like_count"]),
            "favourite_count": int(request.form["favourite_count"]),
            "comment_count": int(request.form["comment_count"]),
        }

        poster_file = request.files["poster"]
        poster_path = None
        if poster_file:
            filename = secure_filename(poster_file.filename)
            poster_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            poster_file.save(poster_path)

       
        generic_input = preprocess_genre_input(data["budget"], data["runtime"], data["view_count"], 
                                               data["like_count"], data["favourite_count"], data["comment_count"], 
                                               data["release_month"], data["genre"])
        
        title_overview_input = preprocess_title_overview_input(data["title"], data["overview"])
         

        generic_pred = generic_model.predict(generic_input)
        prediction = generic_pred[0]

    return render_template("index.html", genres=GENRE_CHOICES, months=MONTH_CHOICES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
