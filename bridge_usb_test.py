import time
import serial

PORT = "/dev/ttyUSB0"
BAUDRATE = 115200


def send_cmd(ser, cmd, wait=0.5):
    print(f"\nTX: {cmd}")
    ser.write((cmd + "\n").encode("ascii"))
    time.sleep(wait)

    rx = ser.read_all()
    if rx:
        print("RX:")
        print(rx.decode(errors="replace"))
    else:
        print("RX: <no data>")


def main():
    print(f"Open serial: {PORT}, baudrate={BAUDRATE}")

    with serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        timeout=0.8,
        write_timeout=1.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    ) as ser:

        print("Wait robot bridge boot...")
        time.sleep(2.5)

        boot_msg = ser.read_all()
        if boot_msg:
            print("BOOT MSG:")
            print(boot_msg.decode(errors="replace"))

        send_cmd(ser, "PING")
        send_cmd(ser, "STOP")

        input("\nSafe, press ENTER to test MOVE_TEST...")
        send_cmd(ser, "MOVE_TEST", wait=1.5)

        send_cmd(ser, "STOP")

        input("\nSafe, press ENTER to test ACTION_0...")
        send_cmd(ser, "ACTION_0", wait=1.0)

        input("\nSafe, press ENTER to test ACTION_1...")
        send_cmd(ser, "ACTION_1", wait=1.0)

        input("\nSafe, press ENTER to test ACTION_2...")
        send_cmd(ser, "ACTION_2", wait=1.0)

        input("\nSafe, press ENTER to test ACTION_3...")
        send_cmd(ser, "ACTION_3", wait=1.0)


if __name__ == "__main__":
    main()