import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # --- ส่วนประกอบหลัก ---
        # 1. LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True  # <--- สำคัญมาก! ทำให้ Tensor มีมิติเป็น (batch, seq_len, features)
        )

        # 2. Fully Connected (Linear) Layer
        self.linear = nn.Linear(
            in_features=hidden_layer_size,
            out_features=output_size
        )

    def forward(self, input_seq):
        # สร้าง Hidden State และ Cell State เริ่มต้นให้เป็นศูนย์
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        # ส่งข้อมูลเข้า LSTM Layer
        # lstm_out จะเก็บ output ของทุกๆ time step
        # (self.hidden_cell) จะเก็บ h_n และ c_n ของ time step สุดท้าย
        lstm_out, _ = self.lstm(input_seq, (h0, c0))

        # เราต้องการแค่ output ของ time step สุดท้ายเพื่อไปทำนาย
        # lstm_out[:, -1, :] คือการเลือกข้อมูลทั้งหมดใน batch, ที่ time step สุดท้าย, และทุก hidden features
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions