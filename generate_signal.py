
from scipy import signal
import numpy as np
import time
from datetime import datetime
class SignalGenerator:
    """
    คลาสนี้ใช้สร้างสัญญาณทดสอบ เช่น step, ramp, sine, impulse ฯลฯ
    เพื่อใช้เป็น input ให้กับ process/plant
    """
    def __init__(self, sampling_time=0.1):
        self.sampling_time = sampling_time  # เวลาระหว่างจุดตัวอย่าง (sampling time)

    def unit_step(self, duration, amplitude=1):
        """สร้างสัญญาณ step (คงที่หลังจากเริ่มต้น)"""
        n_samples = int(duration / self.sampling_time)
        return np.ones(n_samples) * amplitude

    def ramp(self, duration, slope=1):
        """สร้างสัญญาณ ramp (เส้นตรงเพิ่มขึ้นเรื่อยๆ)"""
        n_samples = int(duration / self.sampling_time)
        return np.arange(0, n_samples) * slope * self.sampling_time

    def impulse(self, duration, amplitude=1):
        """สร้างสัญญาณ impulse (พุ่ง spike ที่จุดแรกแล้วเป็นศูนย์)"""
        n_samples = int(duration / self.sampling_time)
        signal = np.zeros(n_samples)
        signal[0] = amplitude
        return signal

    def sine(self, duration, amplitude=1, frequency=1):
        """สร้างสัญญาณไซน์ (sine wave)"""
        n_samples = int(duration / self.sampling_time)
        t = np.arange(0, n_samples) * self.sampling_time
        return amplitude * np.sin(2 * np.pi * frequency * t)

    def square(self, duration, amplitude=1, frequency=1, duty=0.5):
        """สร้างสัญญาณ pulse/square wave"""
        n_samples = int(duration / self.sampling_time)
        t = np.arange(0, n_samples) * self.sampling_time
        return amplitude * ((t % (1/frequency)) < duty/frequency).astype(float)

    def parabolic(self, duration, a=1):
        """สร้างสัญญาณพาราโบลา (quadratic/parabolic input)"""
        n_samples = int(duration / self.sampling_time)
        t = np.arange(0, n_samples) * self.sampling_time
        return a * (t**2)

class TestRunner:
    """
    TestRunner: จัดการการรันสัญญาณทดสอบ
    - เลือกสัญญาณหลายแบบได้
    - บันทึกผลเป็นไฟล์ CSV ต่อสัญญาณ
    """
    def __init__(self, modbus, logger, sampling_time=0.1):
        self.modbus = modbus
        self.logger = logger
        self.sampling_time = sampling_time

    def run_test(self, signals, signal_duration=5, sleep_between=2):
        """
        signals: list ของชื่อสัญญาณ ["step", "sine", "impulse", ...]
        signal_duration: ระยะเวลาของแต่ละสัญญาณ
        sleep_between: เวลาพักระหว่างสัญญาณ
        """
        for sig_type in signals:
            print(f"\n🚀 Running test for signal: {sig_type}")

            # Reset Logger
            self.logger.df = self.logger.df.iloc[0:0]

            # สร้างสัญญาณ
            generator = SignalGenerator(duration=signal_duration, sampling_time=self.sampling_time)
            if sig_type == "step":
                t, signal = generator.step()
            elif sig_type == "ramp":
                t, signal = generator.ramp()
            elif sig_type == "impulse":
                t, signal = generator.impulse()
            elif sig_type == "parabolic":
                t, signal = generator.parabolic()
            elif sig_type == "sine":
                t, signal = generator.sine(freq=1.0)
            elif sig_type == "square":
                t, signal = generator.square(freq=1.0)
            elif sig_type == "sawtooth":
                t, signal = generator.sawtooth(freq=1.0)
            else:
                print(f"⚠️ Unknown signal type: {sig_type}")
                continue

            # ส่งสัญญาณ + บันทึกค่า
            for idx, u in enumerate(signal):
                self.modbus.write_holding_register(1025, int(u))
                y = self.modbus.analog_read(1)
                self.logger.add_data_log(
                    ["time", "input", "output", "signal_type"],
                    [[t[idx]], [u], [y], [sig_type]]
                )
                time.sleep(self.sampling_time)

            # Save log
            timestamp = datetime.now().strftime("%H%M%S")
            file_name = f"{sig_type}_{timestamp}.csv"
            self.logger.save_to_csv(file_name, folder_name="test_results")

            # Delay ก่อนสัญญาณต่อไป
            print(f"⏸ Sleep {sleep_between} sec before next signal...")
            time.sleep(sleep_between)