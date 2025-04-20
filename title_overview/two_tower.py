import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pandas as pd

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

class MovieDatasetWithTags(Dataset):
    def __init__(self, texts, tags, targets, tokenizer, tag_vocab, max_text_len=256, max_tag_len=20):
        self.texts = texts
        self.tags = tags
        self.targets = targets
        self.tokenizer = tokenizer
        self.tag_vocab = tag_vocab
        self.max_text_len = max_text_len
        self.max_tag_len = max_tag_len

    def __len__(self):
        return len(self.texts)

    def encode_tags(self, tag_list):
        tag_ids = [self.tag_vocab.get(tag.lower(), self.tag_vocab['[UNK]']) for tag in tag_list]
        tag_ids = tag_ids[:self.max_tag_len]
        tag_ids += [self.tag_vocab['[PAD]']] * (self.max_tag_len - len(tag_ids))
        return torch.tensor(tag_ids, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float)

        text_enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )

        tag_tensor = self.encode_tags(tags)

        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'tags': tag_tensor,
            'target': target
        }

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

df_train['tags'] = df_train['tags'].apply(lambda x: [tag.strip().lower() for tag in x.split(',') if tag.strip()])
df_test['tags'] = df_test['tags'].apply(lambda x: [tag.strip().lower() for tag in x.split(',') if tag.strip()])

train_texts = df_train['title_overview'].tolist()
train_tags = df_train['tags'].tolist()
train_targets = df_train['revenue'].tolist()

test_texts = df_test['title_overview'].tolist()
test_tags = df_test['tags'].tolist()
test_targets = df_test['revenue'].tolist()

tag_vocab = build_tag_vocab(train_tags + test_tags)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = MovieDatasetWithTags(train_texts, train_tags, train_targets, tokenizer, tag_vocab)
test_dataset = MovieDatasetWithTags(test_texts, test_tags, test_targets, tokenizer, tag_vocab)

train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

model = BERTWithTagCNNRegressor(tag_vocab_size=len(tag_vocab)).to(device)

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=3e-5)

best_val_r2 = 0

epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_preds, train_targets_list = [], []

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags = batch['tags'].to(device)
        target = batch['target'].to(device).unsqueeze(-1)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask, tags)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_preds.extend(output.detach().squeeze().cpu().tolist())
        train_targets_list.extend(target.squeeze().cpu().tolist())

    train_preds = np.expm1(train_preds)
    train_targets = np.expm1(train_targets_list)
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
    train_r2 = r2_score(train_targets, train_preds)

    model.eval()
    with torch.no_grad():
        val_preds, val_targets_list = [], []
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tags = batch['tags'].to(device)
            target = batch['target'].to(device).unsqueeze(-1)

            output = model(input_ids, attention_mask, tags)
            val_preds.extend(output.squeeze().cpu().tolist())
            val_targets_list.extend(target.squeeze().cpu().tolist())

    val_preds = np.expm1(val_preds)
    val_targets = np.expm1(val_targets_list)
    
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_r2 = r2_score(val_targets, val_preds)

    print(f"Epoch {epoch+1} - Training RMSE: ${train_rmse/1000000:.2f}M - Training R Squared: {train_r2:.4f}")
    print(f"Epoch {epoch+1} - Validation RMSE: ${val_rmse/1000000:.2f}M - Validation R Squared: {val_r2:.4f}")

    if val_r2 > best_val_r2 and epoch > 5:
        best_val_r2 = val_r2
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, 'title_overview_two_tower_model.pt')
        print(f"Model saved at epoch {epoch+1} with validation R2: {val_r2:.4f}")

checkpoint = torch.load('title_overview_two_tower_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tags = batch['tags'].to(device)
        targets = batch['target'].cpu().numpy()
        
        outputs = model(input_ids, attention_mask, tags).squeeze().cpu().numpy()
        
        predictions.extend(outputs)
        actuals.extend(targets)

predictions = np.expm1(predictions)
actuals = np.expm1(actuals)

rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"\nFinal Test RMSE: ${rmse/1000000:.2f}M")
print(f"Final Test R²: {r2:.4f}")