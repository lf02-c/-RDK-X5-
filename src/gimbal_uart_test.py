#!/usr/bin/env python3
import serial
import struct
import time

PORT = "/dev/ttyS3"
BAUD = 115200

CMD_ENABLE = 0x01
CMD_DISABLE = 0x02
CMD_SPEED_CTRL = 0x04
CMD_ANGLE_CTRL = 0x05
CMD_LOW_SPEED_CTRL = 0x06


def crc8(data: bytes) -> int:
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc & 0xFF


def packet(cmd: int, yaw: float = 0.0, pitch: float = 0.0) -> bytes:
    data = struct.pack("<Bff", cmd, float(yaw), float(pitch))
    return data + bytes([crc8(data)])


def send(ser, name, cmd, yaw=0.0, pitch=0.0, delay=1.0):
    p = packet(cmd, yaw, pitch)
    ser.write(p)
    ser.flush()
    print(f"[SEND] {name}: cmd=0x{cmd:02X}, yaw={yaw}, pitch={pitch}, hex={p.hex()}")
    time.sleep(delay)


def main():
    print(f"open {PORT} {BAUD}")

    with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
        time.sleep(1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        send(ser, "ENABLE", CMD_ENABLE, 0, 0, 1.0)

        send(ser, "LOW_SPEED yaw +1", CMD_LOW_SPEED_CTRL, 1.0, 0.0, 2.0)
        send(ser, "STOP", CMD_LOW_SPEED_CTRL, 0.0, 0.0, 1.0)

        send(ser, "LOW_SPEED pitch +1", CMD_LOW_SPEED_CTRL, 0.0, 1.0, 2.0)
        send(ser, "STOP", CMD_LOW_SPEED_CTRL, 0.0, 0.0, 1.0)

        send(ser, "ANGLE 0 0", CMD_ANGLE_CTRL, 0.0, 0.0, 2.0)
        send(ser, "STOP", CMD_LOW_SPEED_CTRL, 0.0, 0.0, 0.5)

    print("done")


if __name__ == "__main__":
    main()
