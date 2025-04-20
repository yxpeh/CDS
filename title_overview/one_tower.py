import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd

class MovieDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=256):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        target = self.targets[index]
        encoding = self.tokenizer(text,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target': torch.tensor(target, dtype=torch.float)
        }
    
class BERTRegressor(nn.Module):
    def __init__(self, dropout=0.3):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        return self.regressor(cls_output)

df_train = pd.read_csv('movie_data_train.csv')

df_train['title_overview'] = df_train['original_title'] + ': ' + df_train['overview']

df_train = pd.DataFrame({
    'title_overview': df_train['title_overview'],
    'tags': df_train['tags'].fillna(''),
    'revenue': df_train['revenue']
})

df_test = pd.read_csv('movie_data_test.csv')

df_test['title_overview'] = df_test['original_title'] + ': ' + df_test['overview']

df_test = pd.DataFrame({
    'title_overview': df_test['title_overview'],
    'tags': df_test['tags'].fillna(''),
    'revenue': df_test['revenue']
})

df_train['revenue'] = np.log1p(df_train['revenue'])
df_test['revenue'] = np.log1p(df_test['revenue'])
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

df_train['combined_text'] = df_train['title_overview'] + ' ' + df_train['tags']
df_test['combined_text'] = df_test['title_overview'] + ' ' + df_test['tags']

train_texts = df_train['combined_text'].tolist()
test_texts = df_test['combined_text'].tolist()
train_targets = df_train['revenue'].tolist()
test_targets = df_test['revenue'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = MovieDataset(train_texts, train_targets, tokenizer)
test_dataset = MovieDataset(test_texts, test_targets, tokenizer)

train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

model = BERTRegressor().to(device)
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=3e-5)

epochs = 50
best_val_r2 = 0

train_losses = []
val_rmses = []
val_r2s = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_predictions = []
    train_targets_list = []

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)

        outputs = model(input_ids, attention_mask).squeeze()
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_predictions.extend(outputs.detach().cpu().numpy())
        train_targets_list.extend(targets.detach().cpu().numpy())
    
    train_predictions = np.expm1(train_predictions)
    train_targets = np.expm1(train_targets_list)
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    r2 = r2_score(train_targets, train_predictions)
    rmse = np.sqrt(mean_squared_error(train_targets, train_predictions))

    print(f"\nEpoch {epoch+1}\nTraining Loss: {avg_train_loss:.4f} - Training RMSE: ${rmse/1000000:.2f}M - Training R Squared: {r2:.4f}")

    model.eval()
    val_loss = 0
    val_predictions = []
    val_actuals = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(input_ids, attention_mask).squeeze().cpu().numpy()
            val_predictions.extend(outputs)
            val_actuals.extend(targets.cpu().numpy())
            
            val_loss += criterion(torch.tensor(outputs), targets.cpu()).item()

    val_predictions = np.array(val_predictions)
    val_actuals = np.array(val_actuals)
    
    val_predictions_original = np.expm1(val_predictions)
    val_actuals_original = np.expm1(val_actuals)
    
    val_rmse = np.sqrt(mean_squared_error(val_actuals_original, val_predictions_original))
    val_r2 = r2_score(val_actuals_original, val_predictions_original)
    
    val_rmses.append(val_rmse)
    val_r2s.append(val_r2)
    
    print(f"Epoch {epoch+1} - Validation RMSE: ${val_rmse/1000000:.2f}M - Validation R Squared: {val_r2:.4f}")
    
    if val_r2 >= best_val_r2 and epoch > 5:
        best_val_r2 = val_r2
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
        }, 'title_overview_one_tower_model.pt')
        print(f"Saving model at epoch {epoch+1} at validation R2: {val_r2:.4f}")

checkpoint = torch.load('title_overview_one_tower_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        outputs = model(input_ids, attention_mask).squeeze().cpu().numpy()
        predictions.extend(outputs)
        actuals.extend(targets.cpu().numpy())

predictions = np.expm1(predictions)
actuals = np.expm1(actuals)

rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"Final Test RMSE: ${rmse/1000000:.2f}M - Final Test R²: {r2:.4f}")