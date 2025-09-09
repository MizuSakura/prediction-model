# 1. นำเข้าคลาสที่จำเป็นจากไฟล์ต่างๆ
from comunucation import ModbusTCP
from Logging_andplot import Logger
from generate_signal import TestRunner
import sys

# ==============================================================================
# ---> ส่วนที่ต้องแก้ไข <---
# ==============================================================================
# 1. กำหนดค่า IP และ Port ของอุปกรณ์ PLC หรือ Modbus Server ของคุณ
HOST_IP = '192.168.1.100'
HOST_PORT = 502

# 2. เลือกว่าจะทดสอบด้วยสัญญาณอะไรบ้าง (เลือกจาก: "step", "ramp", "sine", "square" ฯลฯ)
SIGNALS_TO_RUN = ["step", "sine", "ramp"]

# 3. ตั้งค่าการทดสอบ
SAMPLING_TIME = 0.1      # ความเร็วในการเก็บข้อมูล (0.1 = 10 ครั้ง/วินาที)
SIGNAL_DURATION = 10     # ความยาวของแต่ละสัญญาณ (หน่วย: วินาที)
SLEEP_BETWEEN = 3        # เวลาพักระหว่างเปลี่ยนสัญญาณ (หน่วย: วินาที)
# ==============================================================================


def main():
    """
    ฟังก์ชันหลักในการควบคุมกระบวนการทั้งหมด
    """
    # เตรียม object สำหรับบันทึกข้อมูล (Logger) และการสื่อสาร (ModbusTCP)
    logger = Logger()
    modbus = ModbusTCP(host=HOST_IP, port=HOST_PORT)

    try:
        # -- ขั้นตอนที่ 1: เชื่อมต่อกับอุปกรณ์ --
        print(f"กำลังเชื่อมต่อกับ Modbus Server ที่ {HOST_IP}:{HOST_PORT}...")
        if not modbus.connect():
            print("❌ เชื่อมต่อล้มเหลว! กรุณาตรวจสอบ IP Address หรือสาย LAN")
            sys.exit() # ออกจากโปรแกรมทันทีถ้าเชื่อมต่อไม่ได้
        print("✅ เชื่อมต่อสำเร็จ!")

        # -- ขั้นตอนที่ 2: สร้างตัวสั่งงาน TestRunner --
        #    (ส่ง modbus และ logger เข้าไปให้ TestRunner รู้จัก)
        test_runner = TestRunner(modbus=modbus, logger=logger, sampling_time=SAMPLING_TIME)

        # -- ขั้นตอนที่ 3: เริ่มการทดสอบ --
        print(f"\n🚀 เริ่มการทดสอบด้วยสัญญาณ: {', '.join(SIGNALS_TO_RUN)}")
        test_runner.run_test(
            signals=SIGNALS_TO_RUN,
            signal_duration=SIGNAL_DURATION,
            sleep_between=SLEEP_BETWEEN
        )
        print("\n🎉 การทดสอบเสร็จสมบูรณ์!")
        print("ไฟล์ผลลัพธ์ถูกบันทึกในโฟลเดอร์ 'test_results'")

    except Exception as e:
        print(f"เกิดข้อผิดพลาดร้ายแรงระหว่างทำงาน: {e}")

    finally:
        # -- ขั้นตอนที่ 4: ปิดการเชื่อมต่อเสมอ --
        #    (ไม่ว่าจะทำงานสำเร็จหรือล้มเหลว ก็ต้องปิดการเชื่อมต่อ)
        if modbus.client.connected:
            modbus.disconnect()
            print("🔌 ปิดการเชื่อมต่อ Modbus เรียบร้อยแล้ว")


# สั่งให้ฟังก์ชัน main() เริ่มทำงานเมื่อรันไฟล์นี้
if __name__ == "__main__":
    main()