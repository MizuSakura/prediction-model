import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import io

# --- 🧠 0. ส่วนของการจำลองข้อมูล (Data Simulation Backend) ---
# นำ Logic จากไฟล์ simulation_data.py ของคุณมาใส่ไว้ที่นี่
#--------------------------------------------------------------------
class SignalGenerator:
    def __init__(self, t_end=100, dt=0.01):
        self.t = np.arange(0, t_end, dt)
    def pwm(self, amplitude=1, freq=1, duty=0.5):
        T = 1 / freq
        return amplitude * ((self.t % T) < duty * T)

def generate_simulation_data(sim_time=1000, volt_supply=24, dt=0.01, pwm_freq=0.1, pwm_duty=0.5):
    """สร้างข้อมูลจำลองวงจร RC และคืนค่าเป็น DataFrame"""
    R, C = 1.0, 1.0  # สมมติค่า R และ C
    voltage_capacitor = 0.0
    
    sg = SignalGenerator(t_end=sim_time, dt=dt)
    data_input_signal = sg.pwm(amplitude=1, freq=pwm_freq, duty=pwm_duty)
    
    data_output_voltage = []
    
    for signal in data_input_signal:
        v_in = volt_supply * signal
        # สมการของวงจร RC
        v_dot = (v_in - voltage_capacitor) / (R * C)
        voltage_capacitor += v_dot * dt
        data_output_voltage.append(voltage_capacitor)
        
    df = pd.DataFrame({
        'DATA_INPUT': data_input_signal * volt_supply, # PWM signal scaled by voltage
        'DATA_OUTPUT': data_output_voltage
    })
    return df

# --- 🧠 1. ส่วนของโมเดล AI และการจัดการข้อมูล (Backend API) ---
# (ส่วนนี้ยังคงเหมือนเดิม ไม่มีการเปลี่ยนแปลง)
#--------------------------------------------------------------------
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
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesPredictor:
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()

    def create_dataset(self, df):
        data = df[['DATA_INPUT', 'DATA_OUTPUT']].values
        data_scaled = self.scaler_input.fit_transform(data)
        self.scaler_output.fit(df[['DATA_OUTPUT']])

        X, y = [], []
        for i in range(len(data_scaled) - self.params['window_size']):
            X.append(data_scaled[i:i + self.params['window_size']])
            y.append(data_scaled[i + self.params['window_size'], 1])
        return np.array(X), np.array(y).reshape(-1, 1)

    def train(self, df):
        X, y = self.create_dataset(df)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        self.X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        self.model = LSTMModel(input_size=self.params['input_size'], hidden_size=self.params['hidden_size'], num_layers=self.params['num_layers'], output_size=self.params['output_size']).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        st.write("#### 📈 Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_history = []
        self.model.train()
        for epoch in range(self.params['epochs']):
            epoch_loss = 0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            progress_bar.progress((epoch + 1) / self.params['epochs'])
            status_text.text(f'Epoch [{epoch+1}/{self.params["epochs"]}], Loss: {avg_loss:.6f}')
        status_text.success('✅ Training Completed!')
        return loss_history

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(self.X_test_tensor).cpu().numpy()
        predictions = self.scaler_output.inverse_transform(predictions_scaled).flatten()
        return predictions

    def evaluate(self, df):
        _, y = self.create_dataset(df)
        train_size = int(len(y) * 0.8)
        y_test_scaled = y[train_size:]
        actual = self.scaler_output.inverse_transform(y_test_scaled).flatten()
        predicted = self.predict()
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        return {"actual": actual, "predicted": predicted, "mse": mse, "rmse": rmse, "mae": mae}

# --- 🎨 2. ส่วนของ UI ที่แสดงผลและตอบโต้กับผู้ใช้ (Frontend) ---
# (มีการปรับปรุง UI ส่วน Sidebar)
#----------------------------------------------------------------
st.set_page_config(page_title="AI Future Predictor", page_icon="🤖", layout="wide")
st.markdown("""<style> .main .block-container {padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem;} h1, h2, h3, h4 {color: #3Q3Q3Q;} .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 24px; border: none; font-size: 16px;} .stButton>button:hover {background-color: #45a049;} </style>""", unsafe_allow_html=True)

if 'results' not in st.session_state: st.session_state['results'] = None
if 'df_processed' not in st.session_state: st.session_state['df_processed'] = None

# --- Sidebar: ส่วนควบคุม ---
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735319ff6353326543.gif", width=80)
    st.title("⚙️ Control Panel")
    st.markdown("---")

    # --- ✨ NEW: ส่วนจำลองข้อมูล ---
    st.header("1. Choose Data Source")
    source_option = st.radio("เลือกแหล่งข้อมูล:", ('Upload File', 'Simulate Data'), horizontal=True)
    
    df_upload = None
    input_col, output_col = None, None

    if source_option == 'Upload File':
        uploaded_file = st.file_uploader("เลือกไฟล์ CSV, Excel หรือ TXT", type=['csv', 'xlsx', 'txt'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
                
                # --- ✨ NEW: ส่วนเลือกคอลัมน์ ---
                st.write("**Select Columns for Model:**")
                columns = df_upload.columns.tolist()
                input_col = st.selectbox("Select Input Column (X)", columns, index=0)
                output_col = st.selectbox("Select Output Column (Y)", columns, index=min(1, len(columns)-1))

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
                df_upload = None

    else: # Simulate Data
        st.subheader("Simulation Parameters")
        sim_time = st.slider("Simulation Time (seconds)", 100, 5000, 1000)
        pwm_freq = st.slider("PWM Frequency (Hz)", 0.01, 1.0, 0.1, step=0.01)
        pwm_duty = st.slider("PWM Duty Cycle (%)", 0.0, 1.0, 0.5)

    st.markdown("---")

    # --- ส่วน Model Parameters ---
    st.header("2. Configure AI Model")
    window_size = st.slider("Window Size (Timesteps)", 10, 100, 30)
    hidden_size = st.slider("Hidden Layer Size", 10, 200, 100)
    num_layers = st.slider("Number of Layers", 1, 5, 2)
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    epochs = st.slider("Training Epochs", 10, 200, 50)
    
    st.markdown("---")
    train_button = st.button("🚀 Start Training & Predict")


# --- Main Panel: ส่วนแสดงผล ---
st.title("🤖 AI-Powered Time Series Predictor")
st.markdown("แอปพลิเคชันสำหรับ **Train** โมเดล AI เพื่อ **Predict** ค่าในอนาคตจากข้อมูลของคุณ")
st.markdown("---")

df_for_training = None

# --- ตรวจสอบและเตรียมข้อมูลก่อน Train ---
if source_option == 'Upload File':
    if df_upload is not None and input_col and output_col:
        if input_col == output_col:
            st.warning("Input และ Output Column ต้องไม่ซ้ำกัน กรุณาเลือกใหม่")
        else:
            # สร้าง DataFrame ใหม่สำหรับ Train โดยเฉพาะ
            df_for_training = df_upload[[input_col, output_col]].rename(columns={input_col: 'DATA_INPUT', output_col: 'DATA_OUTPUT'})
            st.session_state['df_processed'] = df_for_training
            st.success(f"ไฟล์ '{uploaded_file.name}' พร้อมใช้งานแล้ว")
            st.write("#### 📋 Data Preview (for Training)")
            st.dataframe(df_for_training.head())
    else:
        st.info("กรุณาอัปโหลดไฟล์และเลือกคอลัมน์")

else: # Simulate Data
    if st.button("🔄 Generate & Preview Data"):
        with st.spinner("กำลังสร้างข้อมูลจำลอง..."):
            df_for_training = generate_simulation_data(sim_time, pwm_freq=pwm_freq, pwm_duty=pwm_duty)
            st.session_state['df_processed'] = df_for_training
            st.success("สร้างข้อมูลจำลองสำเร็จแล้ว")
    
    if st.session_state['df_processed'] is not None and source_option == 'Simulate Data':
        df_for_training = st.session_state['df_processed']
        st.write("#### 📋 Simulated Data Preview")
        st.dataframe(df_for_training.head())
        
        st.write("#### 📈 Simulated Data Plot")
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(df_for_training['DATA_OUTPUT'], label='Simulated Output (Capacitor Voltage)')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("กดปุ่ม 'Generate & Preview Data' เพื่อสร้างข้อมูล หรือกด 'Start Training' เพื่อใช้ข้อมูลที่สร้างไว้ล่าสุด")


# --- กระบวนการเมื่อกดปุ่ม Train ---
if train_button:
    # ดึงข้อมูลที่เตรียมไว้ล่าสุดจาก session state
    df_to_use = st.session_state.get('df_processed')

    if df_to_use is not None:
        with st.spinner('กำลังประมวลผลข้อมูลและฝึกโมเดล... กรุณารอสักครู่...'):
            params = {
                'window_size': window_size, 'input_size': 2, 'hidden_size': hidden_size,
                'num_layers': num_layers, 'output_size': 1, 'learning_rate': learning_rate,
                'epochs': epochs, 'batch_size': 64
            }
            predictor = TimeSeriesPredictor(params)
            predictor.train(df_to_use)
            results = predictor.evaluate(df_to_use)
            st.session_state['results'] = results
    else:
        st.warning("ยังไม่มีข้อมูลสำหรับ Train กรุณาอัปโหลดไฟล์หรือสร้างข้อมูลจำลองก่อน")


# --- แสดงผลลัพธ์หลังจาก Train เสร็จ (ส่วนนี้เหมือนเดิม) ---
if st.session_state['results'] is not None:
    results = st.session_state['results']
    st.markdown("---")
    st.header("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{results['mse']:.4f}")
    col2.metric("RMSE", f"{results['rmse']:.4f}")
    col3.metric("MAE", f"{results['mae']:.4f}")

    st.write("#### 📉 Actual vs. Predicted Values")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(results['actual'], label='Actual Data', color='blue', linewidth=2)
    ax.plot(results['predicted'], label='Predicted Data', color='orange', linestyle='--', linewidth=2)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.write("#### 💾 Download Results")
    results_df = pd.DataFrame({'Actual_Output': results['actual'], 'Predicted_Output': results['predicted']})

    @st.cache_data
    def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
    @st.cache_data
    def convert_df_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        return output.getvalue()

    csv_data = convert_df_to_csv(results_df)
    excel_data = convert_df_to_excel(results_df)

    d_col1, d_col2, d_col3 = st.columns([1.5, 1.5, 5])
    with d_col1:
        st.download_button(label="📥 Download as CSV", data=csv_data, file_name='prediction_results.csv', mime='text/csv')
    with d_col2:
        st.download_button(label="📥 Download as Excel", data=excel_data, file_name='prediction_results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')