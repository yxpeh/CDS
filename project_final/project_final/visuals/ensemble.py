import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiImageDataset(Dataset):
    def __init__(self, df, poster_dir, backdrop_dir, thumbnail_dir, transform):
        self.df = df
        self.poster_dir = poster_dir
        self.backdrop_dir = backdrop_dir
        self.thumbnail_dir = thumbnail_dir
        self.transform = transform
        self.valid_ids = []

        for idx, row in df.iterrows():
            movie_id = str(int(row['id']))
            if all(os.path.exists(os.path.join(d, f"{movie_id}.jpg")) for d in [poster_dir, backdrop_dir, thumbnail_dir]):
                self.valid_ids.append(idx)

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        df_idx = self.valid_ids[idx]
        row = self.df.iloc[df_idx]
        movie_id = str(int(row['id']))
        revenue = np.log1p(row['revenue'])

        def load_image(directory):
            image = Image.open(os.path.join(directory, f"{movie_id}.jpg")).convert("RGB")
            return self.transform(image)

        return {
            "poster": load_image(self.poster_dir),
            "backdrop": load_image(self.backdrop_dir),
            "thumbnail": load_image(self.thumbnail_dir),
            "revenue": torch.tensor(revenue, dtype=torch.float)
        }

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

df_train = pd.read_csv("movie_data_train.csv")
df_test = pd.read_csv("movie_data_test.csv")

train_dataset = MultiImageDataset(df_train, "poster_dataset", "backdrop_dataset", "thumbnail_dataset", transform)
test_dataset = MultiImageDataset(df_test, "poster_dataset", "backdrop_dataset", "thumbnail_dataset", transform)

train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(train_dataset, train_idx)
val_subset = torch.utils.data.Subset(train_dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

model = FineTunedEnsemble().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

epochs = 100
best_val_r2 = -np.inf
train_losses, val_r2s = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    preds, targets = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        p = batch["poster"].to(device)
        b = batch["backdrop"].to(device)
        t = batch["thumbnail"].to(device)
        y = batch["revenue"].to(device)

        y_hat = model(p, b, t).squeeze()
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds.extend(y_hat.detach().cpu().view(-1).tolist())
        targets.extend(y.detach().cpu().view(-1).tolist())

    train_losses.append(total_loss / len(train_loader))

    r2_log = r2_score(targets, preds)
    rmse_log = np.sqrt(mean_squared_error(targets, preds))
    r2_real = r2_score(np.expm1(targets), np.expm1(preds))
    rmse_real = np.sqrt(mean_squared_error(np.expm1(targets), np.expm1(preds)))

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Train RMSE: {rmse_log:.4f} | R²: {r2_log:.4f}")

    model.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            p = batch["poster"].to(device)
            b = batch["backdrop"].to(device)
            t = batch["thumbnail"].to(device)
            y = batch["revenue"].to(device)
            y_hat = model(p, b, t).squeeze()
            val_preds.extend(y_hat.cpu().view(-1).tolist())
            val_targets.extend(y.cpu().view(-1).tolist())

    val_r2 = r2_score(np.expm1(val_targets), np.expm1(val_preds))
    val_r2s.append(val_r2)

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        torch.save(model.state_dict(), "best_ensemble_model.pt")
        print(f"Saved new best model with R² = {val_r2:.4f}")

model.load_state_dict(torch.load("best_ensemble_model.pt"))
model.eval()

test_preds, test_targets = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        p = batch["poster"].to(device)
        b = batch["backdrop"].to(device)
        t = batch["thumbnail"].to(device)
        y = batch["revenue"].to(device)
        y_hat = model(p, b, t).squeeze()
        test_preds.extend(y_hat.cpu().view(-1).tolist())
        test_targets.extend(y.cpu().view(-1).tolist())

test_r2 = r2_score(np.expm1(test_targets), np.expm1(test_preds))
test_rmse = np.sqrt(mean_squared_error(np.expm1(test_targets), np.expm1(test_preds)))

print(f"\nTest Results:")
print(f"Test RMSE: ${test_rmse/1e6:.2f}M - Test R²: {test_r2:.4f}")
