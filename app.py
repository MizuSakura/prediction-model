import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import io
import sys
import os
import copy

# ตรวจสอบว่าไฟล์ advanced_model_trainer.py อยู่ในตำแหน่งที่ถูกต้อง
try:
    # สมมติว่าไฟล์ที่คุณให้มาชื่อ GRU_optimized.py
    from GRU_optimized import DataManager, TwinGRU, ModelTrainer
except ImportError:
    st.error("ไม่พบไฟล์ 'GRU_optimized.py' กรุณาตรวจสอบให้แน่ใจว่าไฟล์ app.py และ GRU_optimized.py อยู่ในโฟลเดอร์เดียวกัน")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="TwinGRU Model Evaluator & Comparator",
    page_icon="🧠",
    layout="wide"
)

# --- Helper Functions & Classes ---
def plot_comparison_graph(y_true, y_pred_new, y_pred_loaded=None):
    """สร้างกราฟเปรียบเทียบระหว่างโมเดลใหม่และโมเดลที่โหลดมา"""
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    ax.set_title("Model Comparison: Prediction vs Actual", fontsize=16)
    
    ax.plot(y_true, label="Actual Data", color='black', linewidth=2.5, zorder=5)
    
    if y_pred_new is not None:
        ax.plot(y_pred_new, label="Newly Trained Model", color='red', linestyle='--', alpha=0.9)
    
    if y_pred_loaded is not None:
        ax.plot(y_pred_loaded, label="Loaded Model", color='green', linestyle=':', alpha=0.9, linewidth=2)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    return fig

class StreamlitLog(io.StringIO):
    """
    ดักจับ print statements และแสดงผลใน Streamlit container
    โดยจะแสดงผลทับบรรทัดเดิมเพื่อจำลองการ "refresh"
    """
    def __init__(self, container):
        super().__init__()
        self.container = container

    def write(self, s):
        line = s.strip()
        if line:
            self.container.text(line)

    def flush(self):
        pass

# --- Main App UI ---
st.title("🧠 TwinGRU Model Evaluator & Comparator")
st.markdown("เครื่องมือสำหรับ **ฝึกสอน, ประเมินผล, และเปรียบเทียบ** โมเดล Time Series")

# --- Initialize Session State ---
if 'trained_trainer' not in st.session_state:
    st.session_state['trained_trainer'] = None
if 'loaded_trainer' not in st.session_state:
    st.session_state['loaded_trainer'] = None
if 'data_manager' not in st.session_state:
    st.session_state['data_manager'] = None

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("1. Data Source")
    source = st.radio("เลือกแหล่งข้อมูล:", ("สร้างข้อมูลจำลอง", "อัปโหลดไฟล์ CSV"))
    
    uploaded_file = None
    input_col, output_col = None, None
    if source == "อัปโหลดไฟล์ CSV":
        uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
        if uploaded_file:
            try:
                bytes_data = uploaded_file.getvalue()
                df_preview = pd.read_csv(io.BytesIO(bytes_data), nrows=5)
                cols = df_preview.columns.tolist()
                st.write("ตัวอย่างคอลัมน์:", cols)
                input_col = st.selectbox("เลือกคอลัมน์ Input:", cols)
                output_col = st.selectbox("เลือกคอลัมน์ Output:", cols, index=min(1, len(cols)-1))
            except Exception as e:
                st.error(f"ไม่สามารถอ่านไฟล์ได้: {e}")

    st.header("2. Compare with Saved Model (Optional)")
    uploaded_model_file = st.file_uploader("อัปโหลด Model (.pth)", type=['pth'])
    if uploaded_model_file:
        st.info("ระบบจะค้นหาไฟล์ scaler .pkl ที่มีชื่อเดียวกันโดยอัตโนมัติ")

    st.header("3. Model Hyperparameters")
    n_past = st.slider("Past Steps (n_past)", 10, 100, 30)
    hidden_dim = st.slider("Hidden Dimensions", 10, 200, 50)
    num_layers = st.slider("Number of GRU Layers", 1, 5, 2)

    st.header("4. Training Parameters")
    epochs = st.number_input("Epochs", 10, 1000, 100, 10)
    lr = st.number_input("Learning Rate", 1e-5, 1e-1, 0.001, format="%.4f")

    start_training = st.button("🚀 เริ่มการฝึกโมเดลใหม่", use_container_width=True)

# --- Data Loading and Model Preparation ---
if start_training or (uploaded_model_file and not st.session_state.data_manager):
    try:
        if source == 'อัปโหลดไฟล์ CSV' and uploaded_file:
            # [FIX] บันทึกไฟล์ที่อัปโหลดลงในโฟลเดอร์ชั่วคราว
            # เพื่อให้เราสามารถส่ง file path ที่ถูกต้องไปให้ DataManager ได้
            temp_dir = Path("./temp_data")
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / uploaded_file.name

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ส่ง path ของไฟล์ชั่วคราวไปแทน object 'UploadedFile'
            st.session_state['data_manager'] = DataManager(filepath=temp_file_path, input_col=input_col, output_col=output_col)
            
        elif source == 'สร้างข้อมูลจำลอง':
             st.session_state['data_manager'] = DataManager()
        else:
             st.session_state['data_manager'] = None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        st.session_state['data_manager'] = None

# --- Logic for Loading a Comparison Model ---
if uploaded_model_file and not st.session_state['loaded_trainer']:
    if st.session_state['data_manager']:
        with st.spinner("กำลังโหลดโมเดลสำหรับเปรียบเทียบ..."):
            try:
                loaded_model_instance = TwinGRU(input_dim=1, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
                loaded_trainer = ModelTrainer(model=loaded_model_instance, n_past=n_past)
                
                temp_dir = Path("./temp_models")
                temp_dir.mkdir(exist_ok=True)
                model_path = temp_dir / uploaded_model_file.name
                
                with open(model_path, "wb") as f:
                    f.write(uploaded_model_file.getbuffer())
                
                system_input, system_output = st.session_state['data_manager'].get_data()
                loaded_trainer.prepare_data(system_input, system_output)

                loaded_trainer.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
                loaded_trainer.model.eval()

                st.session_state['loaded_trainer'] = loaded_trainer
                st.sidebar.success(f"โหลดโมเดล '{uploaded_model_file.name}' สำเร็จ")

            except Exception as e:
                st.sidebar.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    else:
        st.warning("กรุณาเลือกแหล่งข้อมูลก่อนทำการโหลดโมเดล")

# --- Logic for Training a New Model ---
if start_training:
    if st.session_state['data_manager']:
        model = TwinGRU(input_dim=1, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
        trainer = ModelTrainer(model=model, n_past=n_past)
        
        system_input, system_output = st.session_state['data_manager'].get_data()
        trainer.prepare_data(system_input, system_output)
        
        st.header("📈 Training Progress")
        log_container = st.empty()
        log_stream = StreamlitLog(log_container)
        
        original_stdout = sys.stdout
        sys.stdout = log_stream
        
        try:
            with st.spinner("กำลังฝึกโมเดล..."):
                trainer.train(epochs=epochs, lr=lr)
            st.success("การฝึกโมเดลเสร็จสิ้น!")
            st.session_state['trained_trainer'] = trainer
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการฝึก: {e}")
        finally:
            sys.stdout = original_stdout
            log_container.empty()
    else:
        st.error("ไม่สามารถเริ่มการฝึกได้: กรุณาเลือกแหล่งข้อมูลให้ถูกต้อง")

# --- Display Results Panel ---
if st.session_state['trained_trainer'] is not None or st.session_state['loaded_trainer'] is not None:
    st.header("📊 ผลลัพธ์และการเปรียบเทียบ")
    
    if st.session_state['trained_trainer']:
        if st.button("💾 Save Newly Trained Model"):
            with st.spinner("Saving model..."):
                trainer_to_save = st.session_state['trained_trainer']
                save_log_container = st.empty()
                log_stream = StreamlitLog(save_log_container)
                original_stdout = sys.stdout
                sys.stdout = log_stream
                try:
                    saved_path = trainer_to_save.save_model()
                    st.success(f"Model saved successfully to: {saved_path}")
                except Exception as e:
                    st.error(f"Failed to save model: {e}")
                finally:
                    sys.stdout = original_stdout
                    save_log_container.empty()

    results = {}
    y_true = None
    
    if st.session_state['trained_trainer']:
        trainer = st.session_state['trained_trainer']
    else:
        trainer = st.session_state['loaded_trainer']
        
    y_true = trainer.scaler_output.inverse_transform(trainer.y_test)

    if st.session_state['trained_trainer']:
        pred = st.session_state['trained_trainer'].predict(st.session_state['trained_trainer'].X_test)
        results['Newly Trained Model'] = {'pred': pred}
    
    if st.session_state['loaded_trainer']:
        pred = st.session_state['loaded_trainer'].predict(st.session_state['loaded_trainer'].X_test)
        results['Loaded Model'] = {'pred': pred}
    
    metrics_data = []
    for name, data in results.items():
        mse = np.mean((y_true - data['pred'])**2)
        metrics_data.append({
            "Model": name,
            "MSE": f"{mse:.4f}",
            "RMSE": f"{np.sqrt(mse):.4f}",
            "MAE": f"{np.mean(np.abs(y_true - data['pred'])):.4f}"
        })
    
    if metrics_data:
        st.write("### 📋 Evaluation Metrics Comparison")
        st.table(pd.DataFrame(metrics_data).set_index('Model'))

    y_pred_new = results.get('Newly Trained Model', {}).get('pred', None)
    y_pred_loaded = results.get('Loaded Model', {}).get('pred', None)
    
    comparison_fig = plot_comparison_graph(y_true, y_pred_new, y_pred_loaded)
    st.pyplot(comparison_fig)

else:
    st.info("กด 'เริ่มการฝึกโมเดลใหม่' หรือ 'อัปโหลดโมเดล' ในแถบด้านข้างเพื่อดูผลลัพธ์")