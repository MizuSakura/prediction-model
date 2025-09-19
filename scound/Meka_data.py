# =============================================================================
# ส่วนที่ 1: Import ไลบรารีที่จำเป็น
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import os

# =============================================================================
# ส่วนที่ 2: สร้าง Custom Dataset สำหรับข้อมูลขนาดใหญ่
# =============================================================================
class BigDataTimeSeriesDataset(Dataset):
    """
    คลาส Dataset ที่ออกแบบมาเพื่อจัดการกับไฟล์ CSV ขนาดใหญ่โดยเฉพาะ
    โดยจะอ่านไฟล์ทีละส่วน (chunk) เพื่อประหยัดหน่วยความจำ
    """
    def __init__(self, csv_path, chunk_size, sequence_length, input_cols, output_col):
        """
        Constructor ของคลาส
        
        พารามิเตอร์:
        - csv_path (str): ที่อยู่ของไฟล์ CSV ขนาดใหญ่
        - chunk_size (int): จำนวนแถวที่จะอ่านเข้ามาในหน่วยความจำในแต่ละครั้ง
        - sequence_length (int): ความยาวของข้อมูลอนุกรมเวลา (window size) ที่จะใช้เป็น input
        - input_cols (list): รายชื่อคอลัมน์ที่จะใช้เป็น Input (เช่น ['DATA_INPUT', 'DATA_OUTPUT'])
        - output_col (str): ชื่อคอลัมน์ที่จะใช้เป็น Output (เช่น 'DATA_OUTPUT')
        """
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.sequence_length = sequence_length
        self.input_cols = input_cols
        self.output_col = output_col

        # สร้าง Scaler สำหรับ Input และ Output แยกกัน เพื่อการแปลงค่ากลับที่ถูกต้อง
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()

        print("กำลังเตรียมข้อมูลและปรับสเกล (Fitting Scaler)...")
        self._fit_scalers()
        
        # คำนวณความยาวทั้งหมดของ Dataset
        # เราจำเป็นต้องรู้จำนวนแถวทั้งหมดเพื่อใช้ใน __len__
        print("กำลังนับจำนวนแถวทั้งหมดในไฟล์...")
        self.total_rows = sum(1 for row in open(self.csv_path)) - 1 # -1 เพื่อไม่นับ header
        print(f"พบข้อมูลทั้งหมด {self.total_rows} แถว")

        self.sequences = []
        self.labels = []
        
    def _fit_scalers(self):
        """
        ฟังก์ชันสำหรับสอน (fit) Scaler โดยการอ่านข้อมูลทั้งหมดแบบ chunk
        เพื่อหาค่า min/max ของแต่ละคอลัมน์โดยไม่โหลดข้อมูลทั้งหมดเข้า RAM
        """
        # สร้าง TextFileReader เพื่อวนลูปอ่านข้อมูลทีละ chunk
        reader = pd.read_csv(self.csv_path, chunksize=self.chunk_size, iterator=True)
        
        start_time = time.time()
        for i, chunk in enumerate(reader):
            # partial_fit จะเรียนรู้ค่า min/max จากข้อมูลทีละส่วน
            self.scaler_input.partial_fit(chunk[self.input_cols])
            self.scaler_output.partial_fit(chunk[[self.output_col]]) # ต้องใส่ใน list 2 ชั้น
            if (i + 1) % 10 == 0:
                 print(f"  - Fit Scaler จาก Chunk ที่ {i+1}...")
        
        end_time = time.time()
        print(f"ปรับสเกลข้อมูลเสร็จสิ้นใน {end_time - start_time:.2f} วินาที")

    def __len__(self):
        """
        ฟังก์ชันพิเศษที่ต้องมีใน Dataset class
        คืนค่าจำนวน sample ทั้งหมดที่โมเดลจะสามารถเรียนรู้ได้
        """
        return self.total_rows - self.sequence_length

    def __getitem__(self, idx):
        """
        ฟังก์ชันพิเศษที่สำคัญที่สุด!
        ทำหน้าที่ดึงข้อมูล 1 sample (sequence และ label) ณ ตำแหน่ง (index) ที่ร้องขอ
        
        พารามิเตอร์:
        - idx (int): ตำแหน่งของ sample ที่ต้องการ
        """
        # คำนวณช่วงของข้อมูลที่ต้องอ่านจากไฟล์ CSV
        start_row = idx
        end_row = idx + self.sequence_length + 1 # +1 เพราะเราต้องการ label ด้วย
        
        # อ่านข้อมูลเฉพาะส่วนที่จำเป็นจากไฟล์ CSV
        # skiprows จะข้ามแถวก่อนหน้าทั้งหมด, nrows จะอ่านเท่าที่จำเป็น
        # นี่คือหัวใจของการประหยัดหน่วยความจำ!
        df_slice = pd.read_csv(
            self.csv_path,
            skiprows=range(1, start_row + 1), # +1 เพราะ skiprows ไม่นับ header
            nrows=self.sequence_length + 1,
            header=None, # ไม่มี header เพราะเราข้ามไปแล้ว
            names=['DATA_INPUT', 'DATA_OUTPUT', 'TIME'] # กำหนดชื่อคอลัมน์เอง
        )
        
        # ดึงข้อมูล Input และ Output
        input_data = df_slice[self.input_cols].values
        output_data = df_slice[[self.output_col]].values
        
        # ปรับสเกลข้อมูลด้วย Scaler ที่ fit ไว้แล้ว
        scaled_input = self.scaler_input.transform(input_data)
        scaled_output = self.scaler_output.transform(output_data)

        # แบ่งข้อมูลเป็น sequence (X) และ label (y)
        sequence = scaled_input[:self.sequence_length]
        label = scaled_output[self.sequence_length]

        # แปลงเป็น PyTorch Tensors
        return torch.FloatTensor(sequence), torch.FloatTensor(label)

# =============================================================================
# ส่วนที่ 3: สร้างโมเดล (ตัวอย่าง LSTM Model)
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # สร้าง hidden state และ cell state เริ่มต้น
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # ส่งข้อมูลเข้า LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # ใช้ output จาก timestep สุดท้ายเท่านั้น
        out = self.fc(out[:, -1, :])
        return out

# =============================================================================
# ส่วนที่ 4: ส่วนควบคุมหลัก (Main Execution)
# =============================================================================
if __name__ == '__main__':
    # --- 1. ตั้งค่า Hyperparameters และค่าต่างๆ ---
    CSV_FILE_PATH = 'data_log_simulation_600K.csv'  # <-- แก้เป็น path ไฟล์ของอาจารย์
    CHUNKSIZE = 50000      # จำนวนแถวที่อ่านในแต่ละครั้ง (ปรับได้ตามขนาด RAM)
    SEQUENCE_LENGTH = 30   # ขนาด Window size
    INPUT_COLUMNS = ['DATA_INPUT', 'DATA_OUTPUT']
    OUTPUT_COLUMN = 'DATA_OUTPUT'
    
    BATCH_SIZE = 128       # ขนาดของ Batch (ปรับได้ตามขนาด VRAM ของ GPU)
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # --- 2. ตรวจสอบว่ามี GPU หรือไม่ และเลือกอุปกรณ์ (Device) ---
    # นี่คือส่วนสำคัญสำหรับ Jetson Nano!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 ใช้อุปกรณ์: {device}")

    # --- 3. สร้าง Instance ของ Dataset และ DataLoader ---
    print("\n--- เริ่มสร้าง Dataset ---")
    full_dataset = BigDataTimeSeriesDataset(
        csv_path=CSV_FILE_PATH,
        chunk_size=CHUNKSIZE,
        sequence_length=SEQUENCE_LENGTH,
        input_cols=INPUT_COLUMNS,
        output_col=OUTPUT_COLUMN
    )

    # แบ่งข้อมูล Train/Test (ตัวอย่างนี้ใช้ 80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    print(f"ขนาดชุดข้อมูลสำหรับ Train: {len(train_dataset)}")
    print(f"ขนาดชุดข้อมูลสำหรับ Test: {len(test_dataset)}")
    
    print("\n--- เริ่มสร้าง DataLoader ---")
    # DataLoader จะดึงข้อมูลจาก Dataset มาสร้างเป็น Batch ให้โดยอัตโนมัติ
    # num_workers > 0 จะใช้หลาย CPU core ช่วยโหลดข้อมูล ทำให้ GPU ไม่ต้องรอ (สำคัญมาก)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, # สลับข้อมูลในแต่ละ epoch เพื่อการเรียนรู้ที่ดีขึ้น
        num_workers=2 # ปรับตามจำนวน core ของ CPU
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    # --- 4. สร้างโมเดล, Loss Function และ Optimizer ---
    model = LSTMModel(input_size=len(INPUT_COLUMNS), hidden_size=50, num_layers=1, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. วนลูปการเทรน ---
    print("\n--- 🚀 เริ่มกระบวนการเทรนโมเดล ---")
    total_training_time = 0
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train() # เข้าสู่โหมดเทรน
        
        # วนลูปเพื่อดึงข้อมูลทีละ Batch จาก DataLoader
        # tqdm เป็น library เสริมเพื่อให้เห็น progress bar สวยงาม (ติดตั้งด้วย pip install tqdm)
        from tqdm import tqdm
        
        for i, (sequences, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            # ส่งข้อมูล Batch ปัจจุบันไปที่ Device (GPU/CPU)
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time += epoch_duration
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, Time: {epoch_duration:.2f} วินาที")

    print(f"\n--- ✅ การเทรนเสร็จสิ้น ---")
    print(f"⏱️ เวลาที่ใช้ในการเทรนทั้งหมด: {total_training_time:.2f} วินาที")
    
    # --- 6. การประเมินผล (Evaluation) ---
    print("\n--- 📊 เริ่มการประเมินผลโมเดล ---")
    model.eval() # เข้าสู่โหมดประเมินผล
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad(): # ไม่คำนวณ gradient เพื่อความเร็ว
        for sequences, labels in tqdm(test_loader, desc="Evaluating"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            
            # ย้ายข้อมูลกลับมาที่ CPU และแปลงเป็น NumPy array เพื่อประเมินผล
            all_predictions.append(outputs.cpu().numpy())
            all_actuals.append(labels.cpu().numpy())

    # รวมผลลัพธ์จากทุก Batch เข้าด้วยกัน
    predictions_flat = np.concatenate(all_predictions).flatten()
    actuals_flat = np.concatenate(all_actuals).flatten()

    # **สำคัญมาก:** แปลงค่าที่ทำนายและค่าจริงกลับเป็นสเกลเดิม
    # เราต้อง reshape ให้เป็น 2D array ก่อนใช้ inverse_transform
    predictions_inversed = full_dataset.scaler_output.inverse_transform(predictions_flat.reshape(-1, 1))
    actuals_inversed = full_dataset.scaler_output.inverse_transform(actuals_flat.reshape(-1, 1))
    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(actuals_inversed, predictions_inversed)
    print(f"ผลการประเมิน (Test Set):")
    print(f"  - Mean Squared Error (MSE): {mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")