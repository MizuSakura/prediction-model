from pymodbus.client import ModbusTcpClient

class ModbusTCP:
    def __init__(self, host='192.168.1.100', port=502):
        self.client = ModbusTcpClient(host=host, port=port)
        self.host = host
        self.port = port

    def connect(self):
        self.client.connect()
        return self.client.connected

    def disconnect(self):
        self.client.close()
        #print("Disconnected from Modbus server.")
        return not self.client.connected

    # -------------------------
    # DIGITAL OUTPUT
    # -------------------------

    # Function Code 01: Read Coil (Digital Output)
    def read_status_output(self, address):
        response = self.client.read_coils(address, 1)
        if response.isError():
            print(f"Error reading digital coil at address {address}")
            return None
        value = response.bits[0]
        #print(f"Read Coil[{address}] = {value}")
        return value

    # Function Code 05: Write Single Coil
    def digital_write(self, address, value):
        response = self.client.write_coil(address, value)
        if response.isError():
            print(f"Error writing digital coil at address {address}")
            return None
        #print(f"Wrote Coil[{address}] = {value}")
        return True

    # Function Code 15: Write Multiple Coils
    def multiple_digital_write(self, address, values=[0, 0, 0, 0]):
        response = self.client.write_coils(address, values)
        if response.isError():
            print(f"Error writing multiple digital coils starting at {address}")
            return None
        #print(f"Wrote Coils[{address} to {address + len(values) - 1}] = {values}")
        return True

    # -------------------------
    # DIGITAL INPUT
    # -------------------------

    # Function Code 02: Read Discrete Input
    def digital_input(self, address, count=1):
        response = self.client.read_discrete_inputs(address=address, count=count)
        if response.isError():
            print(f"Error reading digital input at address {address}")
            return None
        #print(f"Read Discrete Inputs[{address} to {address + count - 1}] = {response.bits}")
        return response.bits[0] if count == 1 else response.bits

    # -------------------------
    # ANALOG INPUT
    # -------------------------

    # Function Code 04: Read Input Register (Analog Input)
    def analog_read(self, address, count=1):
        response = self.client.read_input_registers(address=address, count=count)
        if response.isError():
            print(f"Error reading analog input at address {address}")
            return None
        #print(f"Read Input Registers[{address} to {address + count - 1}] = {response.registers}")
        return response.registers[0] if count == 1 else response.registers

    # -------------------------
    # HOLDING REGISTER (READ/WRITE)
    # -------------------------

    # Function Code 03: Read Holding Registers
    def read_holding_registers(self, address, count=1, slave_id=1):
        response = self.client.read_holding_registers(address=address, count=count, unit=slave_id)
        if response.isError():
            print(f"Error reading holding registers at address {address}")
            return None
        #print(f"Read Holding Registers[{address} to {address + count - 1}] = {response.registers}")
        return response.registers

    # Function Code 06: Write Single Holding Register
    def write_holding_register(self, address, value):
        response = self.client.write_register(address, value)
        if response.isError():
            print(f"Error writing holding register at address {address}")
            return None
        #print(f"Wrote Holding Register[{address}] = {value}")
        return True

    # Function Code 16: Write Multiple Holding Registers
    def multiple_write_holding_registers(self, address, values=[0, 0]):
        response = self.client.write_registers(address, values)
        if response.isError():
            print(f"Error writing multiple holding registers starting at {address}")
            return None
        #print(f"Wrote Holding Registers[{address} to {address + len(values) - 1}] = {values}")
        return True
