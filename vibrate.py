# Importing Libraries
import serial
import time
import threading
arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

def startVibration():
    arduino.write(bytes('1', 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

def stopVibration():
    arduino.write(bytes('0', 'utf-8'))
    print("stop vibrating")

def shortVibration():
    print("start vibrating")
    arduino.write(bytes('1', 'utf-8'))
    timer = threading.Timer(2.5, stopVibration)
    timer.start()
    data = arduino.readline()
    return data