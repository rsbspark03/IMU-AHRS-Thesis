import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# --- Serial setup ---
ser = serial.Serial('COM3', 115200, timeout=1)

# Wait for a valid line
line = ''
while not isinstance(line, list) or len(line) < 13:
    try:
        line = ser.readline().decode(errors='ignore').strip().split(',')
    except:
        continue

# --- Plotting setup ---
queue_length = 50
x = deque(range(1, queue_length + 1), maxlen=queue_length)
y1 = deque(np.zeros(queue_length), maxlen=queue_length)
y2 = deque(np.zeros(queue_length), maxlen=queue_length)
y3 = deque(np.zeros(queue_length), maxlen=queue_length)

fig, ax = plt.subplots()
line1, = ax.plot(x, y1, color='g', label='Roll')
line2, = ax.plot(x, y2, color='r', label='Pitch')
line3, = ax.plot(x, y3, color='b', label='Yaw')

plt.ylim(-180, 180)  # Orientation angles can go from -180 to 180 degrees
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Orientation (degrees)")

# --- Animation update function ---
def update(frame):
    try:
        raw = ser.readline().decode(errors='ignore').strip().split(',')
        if len(raw) >= 13:
            orient = np.array(raw[10:13], dtype=float)

            x.append(x[-1] + 1)
            y1.append(orient[0])
            y2.append(orient[1])
            y3.append(orient[2])

            line1.set_xdata(x)
            line1.set_ydata(y1)
            line2.set_xdata(x)
            line2.set_ydata(y2)
            line3.set_xdata(x)
            line3.set_ydata(y3)

            ax.set_xlim(x[0], x[-1])
    except Exception as e:
        print(f"Read/Parse error: {e}")

# --- Start animation ---
ani = FuncAnimation(fig, update, interval=20)
plt.show()

# Optionally close serial on exit (handle this better in production)
ser.close()
