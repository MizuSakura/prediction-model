# -*- coding: utf-8 -*-
"""
Revised training script for GRU model with sequence-based input and validation set.
สคริปต์ปรับปรุงสำหรับเทรนโมเดล GRU โดยใช้ข้อมูลแบบลำดับ (Sequence) และมีชุดข้อมูลสำหรับวัดผล (Validation)
"""
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # เพิ่มเข้ามาเพื่อแบ่งข้อมูล
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib

# --- การแก้ไข Path สำหรับ Import ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Logging_andplot import Logger
from GRU_predict import GRUModel
from simulation_data import SignalGenerator

# =============================================================================
# ส่วนที่ 1: การเตรียมข้อมูลและเทรนโมเดล (Training)
# =============================================================================

# --- 1.1: คลาสสำหรับจัดการข้อมูล (Dataset) ---
class ActionOutputDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# --- [ใหม่] ฟังก์ชันสำหรับสร้างหน้าต่างข้อมูล (Sliding Window) ---
def create_sequences(inputs, outputs, sequence_length):
    """
    ฟังก์ชันสำหรับแปลงข้อมูลดิบให้เป็นข้อมูลแบบลำดับ (Sequence)
    """
    X, y = [], []
    for i in range(len(inputs) - sequence_length):
        X.append(inputs[i:(i + sequence_length)])
        y.append(outputs[i + sequence_length])
    return np.array(X), np.array(y)


# --- 1.2: การตั้งค่าและโหลดข้อมูล (ปรับปรุงใหม่) ---
PATH_FILE = r"D:\Project_end\prediction_model\scound\data_log_simulation.csv"

# --- Hyperparameters ที่แนะนำให้ปรับจูน ---
SEQUENCE_LENGTH = 20  # [ใหม่] ความยาวของข้อมูลย้อนหลังที่ให้โมเดลดู
HIDDEN_DIM = 256      # [ปรับปรุง] เพิ่มความจุโมเดล
LAYER_DIM = 2
LEARNING_RATE = 5e-4  # [ปรับปรุง] ลด Learning Rate
EPOCHS = 100
BATCH_SIZE = 64
DROPOUT_PROB = 0.1    # [ใหม่] เพิ่ม Dropout เพื่อป้องกัน Overfitting

logger = Logger()
logger.load_csv(path_file=PATH_FILE)
df = logger.df
print("ตัวอย่างข้อมูลจากไฟล์ CSV:")
print(df.head())

DATA_INPUT = logger.result_column("DATA_INPUT")
DATA_OUTPUT = logger.result_column("DATA_OUTPUT")

# --- 1.3: การประมวลผลข้อมูล (ปรับปรุงใหม่) ---
inputs = np.array(DATA_INPUT).reshape(-1, 1)
outputs = np.array(DATA_OUTPUT).reshape(-1, 1)

scaler_input = StandardScaler()
scaler_output = StandardScaler()
inputs_scaled = scaler_input.fit_transform(inputs)
outputs_scaled = scaler_output.fit_transform(outputs)

# --- [ใหม่] สร้างข้อมูลแบบ Sequence ---
X_seq, y_seq = create_sequences(inputs_scaled, outputs_scaled, SEQUENCE_LENGTH)

# --- [ใหม่] แบ่งข้อมูลเป็น Training และ Validation sets ---
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# แปลงเป็น Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# --- 1.4: การสร้าง DataLoader (สำหรับ Train และ Val) ---
train_dataset = ActionOutputDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ActionOutputDataset(X_val_tensor, y_val_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 1.5: การสร้างโมเดล GRU (ปรับปรุง Input Dim) ---
# Input dim ตอนนี้คือจำนวน feature ต่อ 1 timestep (ยังคงเป็น 1)
input_dim = 1
output_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"กำลังใช้งานอุปกรณ์: {device}")

model = GRUModel(input_dim, HIDDEN_DIM, LAYER_DIM, output_dim, DROPOUT_PROB).to(device)

# --- 1.6: การตั้งค่า Loss Function และ Optimizer ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 1.7: Training Loop (ปรับปรุงใหม่) ---
print("\n--- เริ่มการเทรนโมเดล ---")
for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train() # ตั้งเป็นโหมดเทรน
    total_train_loss = 0
    for batch_input, batch_output in train_dataloader:
        batch_input, batch_output = batch_input.to(device), batch_output.to(device)
        
        pred_output = model(batch_input)
        loss = criterion(pred_output, batch_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # --- Validation Phase ---
    model.eval() # ตั้งเป็นโหมดวัดผล
    total_val_loss = 0
    with torch.no_grad(): # ไม่คำนวณ Gradient ในส่วนนี้
        for batch_input, batch_output in val_dataloader:
            batch_input, batch_output = batch_input.to(device), batch_output.to(device)
            pred_output = model(batch_input)
            loss = criterion(pred_output, batch_output)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

print("--- การเทรนเสร็จสิ้น ---\n")