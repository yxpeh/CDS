from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
from utils import preprocess_genre_input, preprocess_title_overview_input, preprocess_poster_input
from werkzeug.utils import secure_filename
import os
import pickle
from transformers import BertTokenizer, BertModel, TimesformerForVideoClassification, AutoImageProcessor

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# GENRE --- GENERIC FEATURES MODEL
generic_model = pickle.load(open("models/generic_features_model.pkl", "rb"))

# TITLE/OVERVIEW --- TWO TOWER BERT MODEL
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TagCNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, num_filters=128, kernel_sizes=(2, 3, 4), dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        conv_outs = [torch.relu(conv(embedded)).max(dim=2)[0] for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        return self.dropout(out)

class BERTWithTagCNNRegressor(nn.Module):
    def __init__(self, tag_vocab_size, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tag_encoder = TagCNNEncoder(tag_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 384, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, text_input_ids, text_attention_mask, tag_token_ids):

        bert_output = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_cls = bert_output.pooler_output  

        tag_feat = self.tag_encoder(tag_token_ids)

        fused = torch.cat([text_cls, tag_feat], dim=1)
        return self.regressor(self.dropout(fused))

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
            "tags": request.form["tags"],
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
        
        title_overview_input = preprocess_title_overview_input(data["title"], data["overview"], data["tags"])
         

        generic_pred = generic_model.predict(generic_input)
        prediction = generic_pred[0]

    return render_template("index.html", genres=GENRE_CHOICES, months=MONTH_CHOICES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
