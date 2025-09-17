import sys
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# --- การแก้ไข Path สำหรับ Import ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Logging_andplot import Logger

# --- Load CSV ---
PATH_FILE = r"D:\Project_end\prediction_model\scound\data_log_simulation.csv"

logger = Logger()
logger.load_csv(path_file=PATH_FILE)
df = logger.df
print("ตัวอย่างข้อมูลจากไฟล์ CSV:")
print(df.head())

DATA_INPUT = logger.result_column("DATA_INPUT")
DATA_OUTPUT = logger.result_column("DATA_OUTPUT")

# --- Hyperparameters ---
WINDOW_SIZE = 30     # จำนวน timestep ย้อนหลังที่ใช้ (เช่น 30 step)
INPUT_SIZE = 2       # DATA_INPUT + DATA_OUTPUT
HIDDEN_SIZE = 100
NUM_LAYERS = 1
OUTPUT_SIZE = 1
LOSS_FUNCTION = nn.MSELoss()
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 50
BATCH_SIZE = 64

# --- เตรียมข้อมูล ---
data_input = np.array(DATA_INPUT).reshape(-1, 1)
data_output = np.array(DATA_OUTPUT).reshape(-1, 1)

scaler_input = MinMaxScaler(feature_range=(-1, 1))
scaler_output = MinMaxScaler(feature_range=(-1, 1))

rescale_input = scaler_input.fit_transform(data_input)
rescale_output = scaler_output.fit_transform(data_output)

# รวมเป็น (input, output)
data_all = np.hstack((rescale_input, rescale_output))

# function create Sequences data for LSTM
def create_sequences(data, window_size=30):
    xs, ys = [], []
    for i in range(len(data) - window_size):
        x = data[i:i+window_size]     # (window_size, input_size)
        y = data[i+window_size][1]    # เอาเฉพาะ output column
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- เตรียม sequence ---
X, y = create_sequences(data_all, WINDOW_SIZE)

# --- Split Data ---
train_size = int(len(y) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"training data: {len(X_train)}")

X_train = torch.from_numpy(X_train).float()              # (batch, seq_len, input_size)
y_train = torch.from_numpy(y_train).float().unsqueeze(1) # (batch, 1)
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float().unsqueeze(1)

# --- DataLoader ---
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])  # เอาเฉพาะ last timestep
        return predictions

# --- สร้างโมเดล ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(TRAINING_EPOCHS):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred = model(X_batch)
        loss = LOSS_FUNCTION(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch:3} | Loss: {loss.item():.6f}')

# --- Evaluation ---
model.eval()
with torch.no_grad():
    X_test, y_test = X_test.to(device), y_test.to(device)
    test_predictions = model(X_test).cpu().numpy()
    y_test = y_test.cpu().numpy()

# --- Inverse Transform ---
actual_shape = (len(y_test), 2)
predicted_shape = (len(test_predictions), 2)

actual_padded = np.zeros(actual_shape)
actual_padded[:, 1] = y_test.flatten()
actual_inversed = scaler_output.inverse_transform(actual_padded)[:, 1]

predicted_padded = np.zeros(predicted_shape)
predicted_padded[:, 1] = test_predictions.flatten()
predicted_inversed = scaler_output.inverse_transform(predicted_padded)[:, 1]

# --- Metrics ---
mse = mean_squared_error(actual_inversed, predicted_inversed)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_inversed, predicted_inversed)

print(f"\n--- ผลการประเมินโมเดล ---")
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')

# --- Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 7))
plt.title('Actual vs. Predicted Output', fontsize=16)
plt.plot(actual_inversed, label='Actual Data', color='blue', linewidth=2.5)
plt.plot(predicted_inversed, label='Predicted Data', color='red', linestyle='--', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('DATA_OUTPUT')
plt.legend()
plt.show()
