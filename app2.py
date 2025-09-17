import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
import io
import os

# --- 🧠 Backend: Simulation & Model ---

class SignalGenerator:
    def __init__(self, t_end=100, dt=0.01): self.t = np.arange(0, t_end, dt)
    def pwm(self, amplitude=1, freq=1, duty=0.5): return amplitude * ((self.t % (1/freq)) < duty * (1/freq))

def generate_simulation_data(sim_time, pwm_freq, pwm_duty):
    R, C, volt_supply, dt = 1.0, 1.0, 24, 0.01
    voltage_capacitor = 0.0
    sg = SignalGenerator(t_end=sim_time, dt=dt)
    data_input_signal = sg.pwm(amplitude=1, freq=pwm_freq, duty=pwm_duty)
    data_output_voltage = []
    for signal in data_input_signal:
        v_in = volt_supply * signal
        v_dot = (v_in - voltage_capacitor) / (R * C)
        voltage_capacitor += v_dot * dt
        data_output_voltage.append(voltage_capacitor)
    return pd.DataFrame({'DATA_INPUT': data_input_signal * volt_supply, 'DATA_OUTPUT': data_output_voltage})

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class TimeSeriesPredictor:
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_size=params['input_size'], hidden_size=params['hidden_size'], num_layers=params['num_layers'], output_size=params['output_size']).to(self.device)
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])

    def create_dataset(self, df):
        data = df[['DATA_INPUT', 'DATA_OUTPUT']].values
        data_scaled = self.scaler_input.fit_transform(data)
        self.scaler_output.fit(df[['DATA_OUTPUT']])
        X, y = [], []
        for i in range(len(data_scaled) - self.params['window_size']):
            X.append(data_scaled[i:i + self.params['window_size']])
            y.append(data_scaled[i + self.params['window_size'], 1])
        return np.array(X), np.array(y).reshape(-1, 1)

    def train(self, df, start_epoch=0, loss_history=[]):
        X, y = self.create_dataset(df)
        train_size = int(len(X) * 0.8)
        X_train, self.X_test = X[:train_size], X[train_size:]
        y_train, self.y_test = y[:train_size], y[train_size:]
        self.X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        criterion = nn.MSELoss()
        
        progress_bar = st.progress(start_epoch / self.params['epochs'], text="Initializing...")
        for epoch in range(start_epoch, self.params['epochs']):
            if st.session_state.get('stop_training', False): return loss_history, "Stopped", epoch
            if st.session_state.get('pause_training', False): return loss_history, "Paused", epoch
            epoch_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            progress_bar.progress((epoch + 1) / self.params['epochs'], text=f'Epoch [{epoch+1}/{self.params["epochs"]}] Loss: {avg_loss:.6f}')
        return loss_history, "Completed", self.params['epochs']

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(self.X_test_tensor).cpu().numpy()
        predicted = self.scaler_output.inverse_transform(predictions_scaled).flatten()
        actual = self.scaler_output.inverse_transform(self.y_test).flatten()
        mse = mean_squared_error(actual, predicted)
        return {"actual": actual, "predicted": predicted, "mse": mse, "rmse": np.sqrt(mse), "mae": mean_absolute_error(actual, predicted)}

# --- 🎨 Frontend: Streamlit UI ---

st.set_page_config(page_title="AI Predictor Pro", page_icon="🏆", layout="wide")

states = {'results': None, 'df_processed': None, 'stop_training': False, 'pause_training': False, 'training_status': 'Not Started', 'training_state': {}}
for key, value in states.items():
    if key not in st.session_state: st.session_state[key] = value

with st.sidebar:
    st.title("🏆 AI Predictor Pro"); st.markdown("---")
    with st.expander("⚙️ General Settings", expanded=True):
        theme = st.selectbox("UI Theme", ["Light", "Dark"])
        default_path = os.path.join(os.getcwd(), "saved_models")
        save_dir = st.text_input("Model Save Directory", default_path)
    
    st.header("1. Data Source")
    source_option = st.radio("เลือกแหล่งข้อมูล:", ('Upload File', 'Simulate Data'), horizontal=True)
    if source_option == 'Upload File':
        uploaded_file = st.file_uploader("เลือกไฟล์ CSV หรือ Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                columns = df_upload.columns.tolist()
                input_col = st.selectbox("Input Column (X)", columns, index=0)
                output_col = st.selectbox("Output Column (Y)", columns, index=min(1, len(columns)-1))
                if input_col and output_col and input_col != output_col:
                    st.session_state['df_processed'] = df_upload[[input_col, output_col]].rename(columns={input_col: 'DATA_INPUT', output_col: 'DATA_OUTPUT'})
            except Exception as e: st.error(f"Error loading file: {e}")
    else:
        sim_time = st.slider("Sim Time", 100, 5000, 1000)
        pwm_freq = st.slider("PWM Freq", 0.01, 1.0, 0.1)
        pwm_duty = st.slider("PWM Duty", 0.0, 1.0, 0.5)
        if st.button("🔄 Generate Data"):
            st.session_state['df_processed'] = generate_simulation_data(sim_time, pwm_freq, pwm_duty)

    st.header("2. AI Model Config")
    window_size = st.slider("Window Size", 10, 100, 30)
    hidden_size = st.slider("Hidden Size", 10, 200, 100)
    num_layers = st.slider("Num Layers", 1, 5, 2)
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    epochs = st.slider("Epochs", 10, 500, 50)
    
    st.header("3. Training Controls")
    control_cols = st.columns(3)
    status = st.session_state.training_status
    if status in ['Not Started', 'Completed', 'Stopped']:
        if control_cols[0].button("🚀 Start", use_container_width=True):
            if st.session_state.df_processed is not None:
                st.session_state.training_status = "Running"; st.session_state.results = None; st.session_state.training_state = {}; st.rerun()
            else: st.warning("No data available.")
    if status == "Running":
        if control_cols[0].button("⏸️ Pause", use_container_width=True): st.session_state.pause_training = True
        if control_cols[2].button("🛑 Stop", use_container_width=True, type="primary"): st.session_state.stop_training = True
    if status == "Paused":
        if control_cols[0].button("▶️ Resume", use_container_width=True): st.session_state.training_status = "Running"; st.session_state.pause_training = False; st.rerun()
        if control_cols[2].button("🛑 Stop", use_container_width=True, type="primary"): st.session_state.stop_training = True

st.title("🏆 AI Time Series Predictor [Pro Version]")
if st.session_state.df_processed is not None: st.dataframe(st.session_state.df_processed.head())

if st.session_state.training_status == "Running":
    if 'predictor' not in st.session_state.training_state:
        params = {'window_size': window_size, 'input_size': 2, 'hidden_size': hidden_size, 'num_layers': num_layers, 'output_size': 1, 'learning_rate': learning_rate, 'epochs': epochs, 'batch_size': 64}
        st.session_state.training_state = {'predictor': TimeSeriesPredictor(params), 'start_epoch': 0, 'loss_history': [], 'params': params}
    
    ts = st.session_state.training_state
    loss_history, status, last_epoch = ts['predictor'].train(st.session_state.df_processed, ts['start_epoch'], ts['loss_history'])
    ts['loss_history'] = loss_history
    
    if status in ["Completed", "Stopped"]:
        if status == "Completed": st.session_state.results = ts['predictor'].evaluate()
        st.session_state.training_status = status; st.session_state.stop_training = False; st.rerun()
    elif status == "Paused":
        st.session_state.training_status = "Paused"; ts['start_epoch'] = last_epoch; st.session_state.pause_training = False; st.rerun()

if st.session_state.results:
    st.markdown("---"); st.header("📊 Prediction Results")
    template = "plotly_dark" if theme == "Dark" else "plotly_white"
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.results['actual'], mode='lines', name='Actual Data'))
    fig.add_trace(go.Scatter(y=st.session_state.results['predicted'], mode='lines', name='Predicted Data', line=dict(dash='dash')))
    fig.update_layout(title='Prediction vs Actual', template=template)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("#### 📋 Evaluation Metrics")
    metrics_df = pd.DataFrame([st.session_state.results]).drop(columns=['actual', 'predicted']).T.rename(columns={0: 'Value'})
    st.table(metrics_df.style.format("{:.4f}"))
    
    with st.expander("📜 View Training Log & Export Model"):
        loss_df = pd.DataFrame({'epoch': range(1, len(st.session_state.training_state['loss_history']) + 1), 'loss': st.session_state.training_state['loss_history']})
        st.line_chart(loss_df.set_index('epoch'))
        
        st.markdown("---")
        st.subheader("Save Trained Model")
        model_name = st.text_input("Model Filename (.pth)", "lstm_model.pth")
        if st.button("📥 Save Model"):
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            save_path = os.path.join(save_dir, model_name)
            try:
                torch.save(st.session_state.training_state['predictor'].model.state_dict(), save_path)
                st.success(f"Model saved successfully to '{save_path}'")
            except Exception as e:
                st.error(f"Failed to save model: {e}")