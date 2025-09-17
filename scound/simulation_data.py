import numpy as np
import matplotlib.pyplot as plt
from Logging_andplot import Logger


class RC_environment:
    def __init__(self, R=1.0, C=1.0, dt=0.01,setpoint = 5,volt_max=24):
        self.R = R  # Resistance in ohms
        self.C = C  # Capacitance in farads
        self.dt = dt  # Time step in seconds
        self.voltage_capacitor = 0.0  # Initial voltage across the capacitor
        

        self.setpoint = setpoint
        self.maximumn_volt = volt_max

        self.time = 0.0
        self.round_reset = 0
        self.per_error = 0
        self.per_action = 0
        self.intergal_error = 0


        self.reset()

    def reset(self,control=None):
        if control is None:
            # min_diff = 0.5 
            # round = 10
            # spread = (self.round_reset / 100) * self.maximumn_volt

            # block = (self.round_reset // round)  

            # if block % 2 == 0:
            #     low = max(0, self.setpoint - spread)
            #     high = self.setpoint - min_diff
            # else:
            #     low = self.setpoint + min_diff
            #     high = min(self.maximumn_volt, self.setpoint + spread)

            self.voltage_capacitor = np.random.uniform(low=0, high=self.maximumn_volt)
        else:
            self.voltage_capacitor = control
            
        self.per_error = self.setpoint - self.voltage_capacitor
        self.round_reset += 1
        self.time = 0.0
        Done = False
        self.per_action = 0

        return self.voltage_capacitor, Done
    
    def step(self, voltage_source=0):
        deltal_volt = (voltage_source - self.voltage_capacitor) / (self.R * self.C)
        self.voltage_capacitor += deltal_volt * self.dt
        self.time += self.dt

        error = self.setpoint - self.voltage_capacitor

        Done = abs(error) <= 0.1

        self.per_error = error
        self.per_action = voltage_source

        return float(self.voltage_capacitor),Done
    
class SignalGenerator:
    def __init__(self, t_end=10, dt=0.01):
        self.t = np.arange(0, t_end, dt)

    def step(self, amplitude=1, start_time=0):
        return (self.t >= start_time) * amplitude

    def impulse(self, amplitude=1, at_time=0):
        signal = np.zeros_like(self.t)
        idx = np.argmin(np.abs(self.t - at_time))
        signal[idx] = amplitude / (self.t[1] - self.t[0])  # scaling ให้ area ≈ amplitude
        return signal

    def pulse(self, amplitude=1, start_time=0, width=1):
        return ((self.t >= start_time) & (self.t <= start_time+width)) * amplitude
    
    def pwm(self, amplitude=1, freq=1, duty=0.5):
        T = 1 / freq
        return amplitude * ((self.t % T) < duty * T)

    def ramp(self, slope=1, start_time=0):
        return slope * np.maximum(0, self.t - start_time)

    def parabolic(self, coeff=1, start_time=0):
        return coeff * np.maximum(0, self.t - start_time)**2
    

TIME_SIMULATION = 6000
VOLT_SUPPLY = 24
DT = 0.01
FOLDER = r"D:\Project_end\prediction_model\scound"
sg = SignalGenerator(t_end= TIME_SIMULATION, dt=DT)
env = RC_environment(volt_max= VOLT_SUPPLY,dt= DT)
logger = Logger()
env.reset()

DATA_INPUT = sg.pwm(amplitude=1,freq=0.1,duty=0.5)
DATA_OUTPUT,ACTION = [],[]
TIME = []

for idx, signal in enumerate(DATA_INPUT):
    output = (VOLT_SUPPLY * signal)
    v_out, done = env.step(voltage_source=output)
    ACTION.append(output)
    DATA_OUTPUT.append(v_out)
    TIME.append(idx)

logger.add_data_log(
        columns_name=["DATA_INPUT", "DATA_OUTPUT"],
        data_list=[ACTION, DATA_OUTPUT])
logger.save_to_csv(file_name="data_log_simulation",folder_name=FOLDER)


plt.plot(TIME,DATA_OUTPUT,label="DATA_OUTPUT")
plt.plot(TIME,ACTION,label="DATA_INPUT SCALE",alpha=0.5)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Test Input Signals")
plt.grid(True)
plt.show()