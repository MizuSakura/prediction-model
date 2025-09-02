import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import joblib
import copy

# =============================================================================
# Class 1: DataManager (มีการเปลี่ยนแปลง)
# =============================================================================
class DataManager:
    """
    คลาสสำหรับจัดการข้อมูล: สร้างข้อมูลจำลอง หรือโหลดจากไฟล์ CSV
    """
    def __init__(self, filepath=None, input_col=None, output_col=None):
        if filepath:
            print(f"Attempting to load data from: {filepath}")
            self._load_from_csv(filepath, input_col, output_col)
        else:
            print("No filepath provided. Generating synthetic data.")
            self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_points=1000):
        # ... (โค้ดส่วนนี้เหมือนเดิม)
        system_input = np.random.choice([-1, 1], size=n_points) * 0.5
        system_input[0:100] = 0.0
        system_input[500:600] = 0.8
        system_output = np.zeros(n_points)
        for t in range(1, n_points):
            system_output[t] = 0.9 * system_output[t-1] + 0.5 * system_input[t-1] + np.random.normal(0, 0.05)
        self.system_input = system_input.reshape(-1, 1)
        self.system_output = system_output.reshape(-1, 1)
        print("Synthetic data generated successfully.")


    def _load_from_csv(self, filepath, input_col, output_col):
        # ... (โค้ดส่วนนี้มีการเปลี่ยนแปลง)
        filepath = Path(filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")
        df = pd.read_csv(filepath)

        # [FIX] เพิ่มโค้ดสำหรับจัดการกับข้อมูลที่หายไป (Missing Values)
        # โดยจะทำการเติมค่า 0 ลงในช่องว่างทั้งหมด
        df.fillna(0, inplace=True)
        print("Filled missing values with 0.")

        if input_col not in df.columns or output_col not in df.columns:
            raise ValueError(f"Columns '{input_col}' or '{output_col}' not found in the CSV file.")
        self.system_input = df[[input_col]].values
        self.system_output = df[[output_col]].values
        print(f"Data loaded successfully from {filepath}.")


    def get_data(self):
        return self.system_input, self.system_output

# =============================================================================
# Class 2: TwinGRU (ไม่มีการเปลี่ยนแปลง)
# =============================================================================
class TwinGRU(nn.Module):
    # ... (โค้ดส่วนนี้เหมือนเดิม)
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(TwinGRU, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

        self.gru2 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_components=False):
        # GRU1
        h1, _ = self.gru1(x)
        out1 = self.fc1(h1[:, -1, :])  # Use the last hidden state

        # GRU2
        h2, _ = self.gru2(x)
        out2 = self.fc2(h2[:, -1, :])

        out = torch.min(out1, out2)
        
        if return_components:
            return out, out1, out2
        return out

# =============================================================================
# Class 3: ModelTrainer (ไม่มีการเปลี่ยนแปลง)
# =============================================================================
class ModelTrainer:
    # ... (โค้ดส่วนนี้เหมือนเดิมทั้งหมด)
    def __init__(self, model: TwinGRU, n_past=20, n_future=1):
        self.model = model
        self.n_past = n_past
        self.n_future = n_future
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.best_model_state = None

    def _create_sequences(self, input_data, output_data):
        X, y = [], []
        for i in range(len(input_data) - self.n_past - self.n_future + 1):
            X.append(input_data[i : i + self.n_past])
            y.append(output_data[i + self.n_past : i + self.n_past + self.n_future])
        return np.array(X), np.array(y).reshape(-1, self.n_future)
        
    def prepare_data(self, system_input, system_output):
        input_scaled = self.scaler_input.fit_transform(system_input)
        output_scaled = self.scaler_output.fit_transform(system_output)
        
        X, y = self._create_sequences(input_scaled, output_scaled)
        
        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        
        print(f"Data prepared: X_train shape={self.X_train.shape}, y_train shape={self.y_train.shape}")
        return self.X_train, self.y_train, self.X_test, self.y_test

    def train(self, epochs=50, lr=0.001):
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor) 
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).numpy()
        return self.scaler_output.inverse_transform(y_pred_scaled)

    def evaluate_and_plot(self):
        y_pred = self.predict(self.X_test)
        y_test_original = self.scaler_output.inverse_transform(self.y_test)
        
        mse = np.mean((y_test_original - y_pred)**2)
        print(f"\nTest MSE: {mse:.6f}")

        plt.figure(figsize=(15, 6))
        plt.title("Model Prediction vs Actual Data (Test Set)")
        plt.plot(y_test_original, label="Actual Output", color='blue', alpha=0.7)
        plt.plot(y_pred, label="Predicted Output", color='red', linestyle='--', alpha=0.8)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _get_unique_filepath(self, filepath: Path) -> Path:
        if not filepath.exists(): return filepath
        counter = 1
        while True:
            new_path = filepath.with_name(f"{filepath.stem}({counter}){filepath.suffix}")
            if not new_path.exists(): return new_path
            counter += 1

    def _default_folder(self) -> Path:
        today = datetime.now().strftime("%d_%m_%Y")
        folder = Path.cwd() / "models" / today
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_model(self, filepath: str = None, overwrite=False):
        filepath = Path(filepath) if filepath else self._default_folder() / "model.pth"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists() and not overwrite:
            filepath = self._get_unique_filepath(filepath)
        print(f"Saving model to {filepath}...")
        torch.save(self.model.state_dict(), filepath)
        joblib.dump(self.scaler_input, filepath.with_suffix(".input_scaler.pkl"))
        joblib.dump(self.scaler_output, filepath.with_suffix(".output_scaler.pkl"))
        print("Model and scalers saved successfully.")
        return filepath

    def load_model(self, filepath: str):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        print(f"Loading model from {filepath}...")
        self.model.load_state_dict(torch.load(filepath, map_location=torch.device("cpu")))
        self.model.eval()
        self.scaler_input = joblib.load(filepath.with_suffix(".input_scaler.pkl"))
        self.scaler_output = joblib.load(filepath.with_suffix(".output_scaler.pkl"))
        print("Model and scalers loaded successfully.")