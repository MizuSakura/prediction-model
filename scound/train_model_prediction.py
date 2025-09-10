from Logging_andplot import Logger
from GRU_predict import GRUModel
from simulation_data import SignalGenerator
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim



class ActionOutputDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

PATH_FILE = fr"D:\Project_end\prediction_model\scound\data_log_simulation.csv"
HIDDEN_DIM = 128
LAYER_DIM = 2
logger = Logger()
logger.load_csv(path_file=PATH_FILE)
df = logger.df
print(df.head())
DATA_INPUT = logger.result_column("DATA_INPUT")
DATA_OUTPUT = logger.result_column("DATA_OUTPUT")

inputs = np.array(DATA_INPUT).reshape(-1, 1)  
outputs = np.array(DATA_OUTPUT).reshape(-1, 1)
scaler_input = StandardScaler()
scaler_output = StandardScaler()
inputs_scaled = scaler_input.fit_transform(inputs)
outputs_scaled = scaler_output.fit_transform(outputs)
inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs_scaled, dtype=torch.float32)

dataset = ActionOutputDataset(inputs_tensor, outputs_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

input_dim = inputs_tensor.shape[1]   
output_dim = outputs_tensor.shape[1] 
dropout_prob = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRUModel(input_dim, HIDDEN_DIM, LAYER_DIM, output_dim, dropout_prob).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10000
for epoch in range(epochs):
    total_loss = 0
    for batch_input, batch_output in dataloader:
        batch_input = batch_input.to(device)
        batch_output = batch_output.to(device)

        pred_output = model(batch_input)

        loss = criterion(pred_output, batch_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

class GRU_Environment:

    def __init__(self, model, scaler_input, scaler_output, dt=0.01, volt_max=24, device="cpu"):
        self.model = model
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output
        self.dt = dt
        self.volt_max = volt_max
        self.device = device

        self.reset()

    def reset(self, control=None):
        # กำหนดค่า output เริ่มต้น (random หรือจาก control)
        if control is None:
            self.v_out = np.random.uniform(0, self.volt_max)
        else:
            self.v_out = control
        self.time = 0
        self.done = False
        return self.v_out, self.done

    def step(self, action):
        # action → scale → tensor
        action_scaled = self.scaler_input.transform(np.array([[action]]))
        action_tensor = torch.tensor(action_scaled, dtype=torch.float32).to(self.device)

        # ใช้ GRU predict next output
        with torch.no_grad():
            pred_scaled = self.model(action_tensor)
        pred_output = self.scaler_output.inverse_transform(pred_scaled.cpu().numpy())

        # อัพเดตสถานะ environment
        self.v_out = pred_output.item()
        self.time += self.dt

        # optional: สร้างเงื่อนไข done เช่น error ใกล้ target
        self.done = False

        return self.v_out, self.done
    
TIME_SIMULATION = 60
VOLT_SUPPLY = 24
DT = 0.01

# ตัวอย่าง signal generator
sg = SignalGenerator(t_end=TIME_SIMULATION, dt=DT)
DATA_INPUT = sg.pwm(amplitude=1, freq=0.1, duty=0.5)

# สร้าง environment
gru_env = GRU_Environment(model=model, scaler_input=scaler_input, scaler_output=scaler_output, dt=DT, volt_max=VOLT_SUPPLY, device=device)
gru_env.reset()

DATA_OUTPUT, ACTION, TIME = [], [], []

for idx, signal in enumerate(DATA_INPUT):
    action = VOLT_SUPPLY * signal
    v_out, done = gru_env.step(action)
    ACTION.append(action)
    DATA_OUTPUT.append(v_out)
    TIME.append(idx)

# plot
import matplotlib.pyplot as plt
plt.plot(TIME, DATA_OUTPUT, label="Predicted OUTPUT (GRU)")
plt.plot(TIME, ACTION, label="Input ACTION", alpha=0.5)
plt.xlabel("Time step")
plt.ylabel("Voltage")
plt.title("GRU Simulation Test")
plt.legend()
plt.grid(True)
plt.show()

# save
torch.save(model.state_dict(), "gru_model_10000_ep.pth")
print("Model saved!")

# load (สร้าง model ใหม่แล้ว load)
model_loaded = GRUModel(input_dim, HIDDEN_DIM, LAYER_DIM, output_dim, dropout_prob).to(device)
model_loaded.load_state_dict(torch.load("gru_model_10000_ep.pth", map_location=device))
model_loaded.eval()  # important: set to evaluation mode

import joblib

joblib.dump(scaler_input, "scaler_input_10000.save")
joblib.dump(scaler_output, "scaler_output_10000.save")

# โหลดกลับ
scaler_input = joblib.load("scaler_input_10000.save")
scaler_output = joblib.load("scaler_output_10000.save")