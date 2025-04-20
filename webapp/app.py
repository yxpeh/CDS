from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
from utils import preprocess_genre_input
from werkzeug.utils import secure_filename
import os
import pickle
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict
import pandas as pd

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

device = torch.device("cpu")

# GENRE --- GENERIC FEATURES MODEL
generic_model = pickle.load(open("models/generic_features_model.pkl", "rb"))

# TITLE/OVERVIEW --- TWO TOWER BERT MODEL
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

def build_tag_vocab(tag_lists, min_freq=1):
    tag_freq = defaultdict(int)
    for tags in tag_lists:
        for tag in tags:
            tag_freq[tag.lower()] += 1

    vocab = {'[PAD]': 0, '[UNK]': 1}
    for tag, freq in tag_freq.items():
        if freq >= min_freq:
            vocab[tag] = len(vocab)

    return vocab

df_train = pd.read_csv('movie_data_train.csv')
df_test = pd.read_csv('movie_data_test.csv')

df_train['tags'] = df_train['tags'].fillna('').apply(
    lambda x: [tag.strip().lower() for tag in x.split(',') if tag.strip()]
)
df_test['tags'] = df_test['tags'].fillna('').apply(
    lambda x: [tag.strip().lower() for tag in x.split(',') if tag.strip()]
)

train_tags = df_train['tags'].tolist()
test_tags = df_test['tags'].tolist()
tag_vocab = build_tag_vocab(train_tags + test_tags)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
title_model = BERTWithTagCNNRegressor(tag_vocab_size=len(tag_vocab)).to(device)
title_model.load_state_dict(torch.load('models/title_overview_two_tower_model.pt', map_location=device)['model_state_dict'])
title_model.eval()

# POSTER --- VISUAL ENSEMBLE MODEL
def get_resnet_backbone():
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    for param in resnet.avgpool.parameters():
        param.requires_grad = True
    return nn.Sequential(*list(resnet.children())[:-1])

class FineTunedEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.poster_net = get_resnet_backbone()
        self.backdrop_net = get_resnet_backbone()
        self.thumbnail_net = get_resnet_backbone()

        self.mlp = nn.Sequential(
            nn.Linear(2048*3, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, poster, backdrop, thumbnail):
        p = self.poster_net(poster)
        b = self.backdrop_net(backdrop)
        t = self.thumbnail_net(thumbnail)
        x = torch.cat([p.view(p.size(0), -1), b.view(b.size(0), -1), t.view(t.size(0), -1)], dim=1)
        return self.mlp(x)

visual_model = FineTunedEnsemble().to(device)
visual_model.load_state_dict(torch.load('models/best_ensemble_model.pt', map_location=device))
visual_model.eval()

# LATE FUSION --- GATING ENSEMBLE MODEL
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 16]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU()
            ])
        layers.append(nn.Linear(dims[-1], input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        weights = F.softmax(logits, dim=1)
        return weights

class GatingFusionEnsemble:
    def __init__(self):
        self.gating_net = None

    def predict(self, X):
        self.gating_net.eval()
        with torch.no_grad():
            weights = self.gating_net(X)
            return torch.sum(X * weights, dim=1, keepdim=True)

gating_checkpoint = torch.load('models/gating_ensemble.pt', map_location=device, weights_only=False)
gating_ensemble = GatingFusionEnsemble()
gating_ensemble.gating_net = GatingNetwork(input_dim=4).to(device)
gating_ensemble.gating_net.load_state_dict(gating_checkpoint['model_state_dict'])
gating_ensemble.gating_net.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
            "tags": request.form["tags"].split(',')
        }

        poster_file = request.files["poster"]
        poster_path = None
        if poster_file:
            filename = secure_filename(poster_file.filename)
            poster_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            poster_file.save(poster_path)
            poster_img = Image.open(poster_path).convert("RGB")
            poster_tensor = transform(poster_img).unsqueeze(0).to(device)
        else:
            poster_tensor = torch.zeros(1, 3, 224, 224).to(device)

        generic_input = preprocess_genre_input(data["budget"], data["runtime"], data["view_count"], 
                                             data["like_count"], data["favourite_count"], data["comment_count"], 
                                             data["release_month"], data["genre"])
        generic_pred = generic_model.predict(generic_input)[0]

        title_overview = data["title"] + ": " + data["overview"]
        text_enc = tokenizer(
            title_overview,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        tag_ids = []
        for tag in data["tags"]:
            tag_ids.append(tag_vocab.get(tag.lower(), tag_vocab['[UNK]']))
        while len(tag_ids) < 20:
            tag_ids.append(tag_vocab['[PAD]'])
        tag_ids = torch.tensor(tag_ids[:20]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            title_pred = title_model(
                text_enc['input_ids'].to(device),
                text_enc['attention_mask'].to(device),
                tag_ids
            ).squeeze().cpu().numpy()

        with torch.no_grad():
            visual_pred = visual_model(poster_tensor, poster_tensor, poster_tensor).squeeze().cpu().numpy()

        all_preds = np.array([
            np.expm1(title_pred),
            np.expm1(visual_pred),
            np.expm1(visual_pred),
            generic_pred
        ]).reshape(1, -1)
        
        all_preds_scaled = gating_checkpoint['X_scaler'].transform(all_preds)
        all_preds_tensor = torch.tensor(all_preds_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            final_pred = gating_ensemble.predict(all_preds_tensor)
            final_pred = gating_checkpoint['y_scaler'].inverse_transform(final_pred.cpu().numpy())
            prediction = final_pred[0][0]

    return render_template("index.html", genres=GENRE_CHOICES, months=MONTH_CHOICES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
