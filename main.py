import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# Custom Classes and Functions
from data_prep import create_sequence
from qmodel import TransformerModel, QTransformerModel, QTransformerModel_ED



date_time_str = datetime.now().strftime("%Hh_%d_%m_%Y")

# Data Preparation

data_dir = 'data'
df = pd.read_csv(f'{data_dir}/am_data.csv')
df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S")
df['year'] = df.date.dt.year.values
df = df[df.cluster == 1]

df_train = df[df['year'] <= 2018]
df_valid = df[df['year'] == 2019]
df_test = df[df['year'] == 2020]

ghi_train = df_train.drop(['date', 'year', 'cluster'], axis=1)
ghi_valid = df_valid.drop(['date', 'year', 'cluster'], axis=1)
ghi_test = df_test.drop(['date', 'year', 'cluster'], axis=1)

cols = list(ghi_train.columns)
scaler = StandardScaler()
ghi_train[cols] = scaler.fit_transform(ghi_train[cols])
ghi_valid[cols] = scaler.fit_transform(ghi_valid[cols])
ghi_test[cols] = scaler.transform(ghi_test[cols])

# Sequence Data Preparation
SEQUENCE_SIZE = 36*2
target_window = 36

print(f'[LOG]: Preparing Dataset')
# Create Sequences
x_train, y_train = create_sequence(SEQUENCE_SIZE, target_window, ghi_train)
x_valid, y_valid = create_sequence(SEQUENCE_SIZE, target_window, ghi_valid)
x_test, y_test = create_sequence(SEQUENCE_SIZE, target_window, ghi_test)

# Setup train data loaders for batch
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Setup valid data loaders for batch
valid_dataset = TensorDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# Setup test data loaders for batch
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = QTransformerModel().to(device)

print(model)


# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                              patience=3, verbose=True)

epochs = 1000
early_stop_count = 0
min_val_loss = float('inf')

tloss = []
vloss = []

for epoch in range(epochs):
    model.train()
    train_loss = []
    print(f'[INFO]: EPOCH {epoch + 1} TRAINING PROCESS')
    for batch in tqdm(train_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    tloss.append(np.mean(train_loss))

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    vloss.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1
        print(f'> [WARNING!]Early Stopping Counter: {early_stop_count}')

    if early_stop_count >= 10:
        print(f"Early stopping! @ epoch {epoch + 1}")
        break

    print(f"[INFO] Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
    print('')


# Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        predictions.extend(outputs.squeeze().tolist())


losses = pd.DataFrame()
losses['train'] = tloss
losses['valid'] = vloss

losses.to_csv(f'results_/losses/{str(date_time_str)}losses_c2qencoderdecoder.csv', index=False)
np.save(f'results_/{str(date_time_str)}__c2truth_QTEncoderDecoder.npy', np.array(y_test))
np.save(f'results_/{str(date_time_str)}__c2preds_QTEncoderDecoder.npy', np.array(predictions))

# torch.save(model, f'models/qtrans-decoder_{date_time_str}')