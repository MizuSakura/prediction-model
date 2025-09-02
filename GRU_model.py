# =============================================================================
# Section 0: Import Libraries
# =============================================================================
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# =============================================================================
# Class 1: DataManager (ไม่มีการเปลี่ยนแปลง)
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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")
        df = pd.read_csv(filepath)
        if input_col not in df.columns or output_col not in df.columns:
            raise ValueError(f"Columns '{input_col}' or '{output_col}' not found in the CSV file.")
        self.system_input = df[[input_col]].values
        self.system_output = df[[output_col]].values
        print(f"Data loaded successfully from {filepath}.")

    def get_data(self):
        return self.system_input, self.system_output

# =============================================================================
# Class 2: GRUModel (ไม่มีการเปลี่ยนแปลง)
# =============================================================================
class GRUModel(nn.Module):
    """
    คลาสสำหรับนิยามสถาปัตยกรรมของโมเดล GRU
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# =============================================================================
# Class 3: ModelTrainer (มีการเปลี่ยนแปลง: เพิ่ม save/load)
# =============================================================================
class ModelTrainer:
    """
    คลาสสำหรับควบคุมกระบวนการฝึกสอน, ประเมินผล, และจัดการโมเดล
    """
    def __init__(self, model, system_input=None, system_output=None, n_past=20, n_future=1):
        self.model = model
        self.system_input = system_input
        self.system_output = system_output
        self.n_past = n_past
        self.n_future = n_future
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self._data_prepared = False # Flag to check if data is ready

    def _prepare_data(self):
        """เตรียมข้อมูล: Scaling และ Windowing"""
        if self.system_input is None or self.system_output is None:
            raise ValueError("System input and output data must be provided for preparation.")
            
        scaled_input = self.scaler_input.fit_transform(self.system_input)
        scaled_output = self.scaler_output.fit_transform(self.system_output)
        scaled_data = np.hstack((scaled_input, scaled_output))
        
        X, y = self._create_sequences(scaled_data)
        
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = y_test
        self._data_prepared = True

    def _create_sequences(self, data):
        """สร้างหน้าต่างข้อมูล (helper function)"""
        X, y = [], []
        for i in range(len(data) - self.n_past - self.n_future + 1):
            X.append(data[i : i + self.n_past])
            y.append(data[i + self.n_past : i + self.n_past + self.n_future, 1])
        return np.array(X), np.array(y)

    def train(self, epochs=50, batch_size=32, lr=0.001):
        """เริ่มกระบวนการฝึกสอนโมเดล"""
        print("--- Starting Training Process ---")
        if not self._data_prepared:
            self._prepare_data()
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(self.X_train_tensor), batch_size):
                X_batch = self.X_train_tensor[i:i+batch_size]
                y_batch = self.y_train_tensor[i:i+batch_size]
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
        print("--- Training Finished ---")

    # ----- ⬇️ เมธอดใหม่สำหรับ SAVE MODEL ⬇️ -----
    def save_model(self, filepath="gru_model.pth"):
        """เซฟพารามิเตอร์ของโมเดล (state_dict) ลงไฟล์"""
        print(f"Saving model to {filepath}...")
        torch.save(self.model.state_dict(), filepath)
        print("Model saved successfully.")

    # ----- ⬇️ เมธอดใหม่สำหรับ LOAD MODEL ⬇️ -----
    def load_model(self, filepath="gru_model.pth"):
        """โหลดพารามิเตอร์ของโมเดล (state_dict) จากไฟล์"""
        print(f"Loading model from {filepath}...")
        # ต้องแน่ใจว่าโมเดลถูกสร้างด้วยสถาปัตยกรรมเดียวกัน
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval() # ตั้งค่าเป็น evaluation mode หลังโหลด
        print("Model loaded successfully.")

    def evaluate_and_plot(self):
        """ประเมินผลโมเดลด้วย Test set และพล็อตกราฟ"""
        print("--- Evaluating Model ---")
        if not self._data_prepared:
            self._prepare_data() # เตรียมข้อมูลถ้ายังไม่ถูกเตรียม

        self.model.eval()
        with torch.no_grad():
            predictions_tensor = self.model(self.X_test_tensor)
        
        predictions_scaled = predictions_tensor.numpy()
        
        predictions_original = self.scaler_output.inverse_transform(predictions_scaled)
        y_test_original = self.scaler_output.inverse_transform(self.y_test)
        
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_original, label='Actual Output', color='blue')
        plt.plot(predictions_original, label='Predicted Output', color='red', linestyle='--')
        plt.title('Model Performance', fontsize=16)
        plt.xlabel('Time Step')
        plt.ylabel('System Output Value')
        plt.legend()
        plt.grid(True)
        plt.show()

# =============================================================================
# Main Execution Block - ส่วนเรียกใช้งาน
# =============================================================================
if __name__ == "__main__":
    
    # --- 1. เตรียมข้อมูล ---
    data_manager = DataManager()
    input_data, output_data = data_manager.get_data()
    
    # --- 2. สร้างโมเดลและ Trainer ---
    MODEL_FILEPATH = "my_trained_gru_model.pth"
    INPUT_SIZE = 2
    HIDDEN_SIZE = 50
    NUM_LAYERS = 1
    OUTPUT_SIZE = 1

    gru_model = GRUModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    trainer = ModelTrainer(model=gru_model, system_input=input_data, system_output=output_data)
    
    # --- 3. ฝึกสอนโมเดล ---
    trainer.train(epochs=100)
    
    # --- 4. ประเมินผลโมเดลที่เพิ่งฝึกเสร็จ ---
    print("\n--- Evaluating the model immediately after training ---")
    trainer.evaluate_and_plot()
    
    # --- 5. เซฟโมเดลที่ฝึกแล้วลงไฟล์ ---
    trainer.save_model(MODEL_FILEPATH)
    
    print("\n" + "="*50 + "\n")
    
    # --- 6. จำลองการใช้งาน: โหลดโมเดลมาใช้ใหม่ ---
    print("### SIMULATING A NEW SESSION: LOADING THE SAVED MODEL ###")
    
    # สร้างโมเดลใหม่ (สถาปัตยกรรมเดิม แต่ยังไม่ผ่านการฝึก)
    new_gru_model = GRUModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    
    # สร้าง Trainer ใหม่สำหรับโมเดลใหม่นี้ (อาจไม่ต้องใส่ข้อมูลถ้าแค่จะโหลด)
    # แต่ต้องใส่ข้อมูลถ้าจะ evaluate เพราะต้องใช้ scaler
    inference_trainer = ModelTrainer(model=new_gru_model, system_input=input_data, system_output=output_data)

    # โหลดค่าน้ำหนักที่เซฟไว้เข้ามาในโมเดลใหม่
    inference_trainer.load_model(MODEL_FILEPATH)
    
    # ประเมินผลโมเดลที่โหลดมา (ควรจะได้ผลลัพธ์เหมือนกับข้างบน)
    print("\n--- Evaluating the loaded model ---")
    inference_trainer.evaluate_and_plot()