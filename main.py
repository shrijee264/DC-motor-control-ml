
import numpy as np
import matplotlib.pyplot as plt

from motor import DCMotor
from pid import PIDController
from signal_processing import add_noise, moving_average


dt = 0.01
duration = 10
setpoint = 1000

time = np.arange(0, duration, dt)

motor = DCMotor()
pid = PIDController(kp=0.008, ki=0.04, kd=0.0001)

speed_values = []
voltage_values = []

for t in time:
    voltage = pid.update(setpoint, motor.speed, dt)
    speed = motor.update(voltage, dt)

    speed_values.append(speed)
    voltage_values.append(voltage)

actual_speed = np.array(speed_values)

np.random.seed(42)
noisy_speed = add_noise(actual_speed, noise_level=10)
filtered_speed = moving_average(noisy_speed, window_size=10)

plt.figure(figsize=(10, 5))
plt.plot(time, actual_speed, label="Actual Speed")
plt.plot(time, noisy_speed, alpha=0.4, label="Noisy Speed")
plt.plot(time, filtered_speed, label="Filtered Speed")
plt.axhline(setpoint, linestyle="--", label="Desired Speed")

plt.xlabel("Time (seconds)")
plt.ylabel("Speed (RPM)")
plt.title("Closed-Loop DC Motor Control")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("motor_response.png", dpi=150)
plt.show()

print(f"Target speed: {setpoint} RPM")
print(f"Final simulated speed: {actual_speed[-1]:.2f} RPM")
print(f"Final control voltage: {voltage_values[-1]:.2f} V")
print("Graph saved as motor_response.png")
