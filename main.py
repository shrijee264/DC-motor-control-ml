
import numpy as np
import matplotlib.pyplot as plt

from motor import DCMotor
from pid import PIDController
from signal_processing import add_noise, moving_average


dt = 0.01
duration = 10
setpoint = 1000
disturbance_time = 5.0
load_torque_value = 0.002

time = np.arange(0, duration, dt)

motor = DCMotor()
pid = PIDController(kp=0.008, ki=0.04, kd=0.0001)

speed_values = []
voltage_values = []

for t in time:
    load_torque = load_torque_value if t >= disturbance_time else 0.0

    voltage = pid.update(setpoint, motor.speed, dt)
    speed = motor.update(voltage, dt, load_torque=load_torque)

    speed_values.append(speed)
    voltage_values.append(voltage)

actual_speed = np.array(speed_values)
voltage_values = np.array(voltage_values)

np.random.seed(42)
noisy_speed = add_noise(actual_speed, noise_level=10)
filtered_speed = moving_average(noisy_speed, window_size=10)

error = setpoint - actual_speed

final_error = abs(error[-1])
overshoot = max(
    0.0,
    (np.max(actual_speed) - setpoint) / setpoint * 100
)

above_10 = np.where(actual_speed >= 0.1 * setpoint)[0]
above_90 = np.where(actual_speed >= 0.9 * setpoint)[0]

if len(above_10) and len(above_90):
    rise_time = time[above_90[0]] - time[above_10[0]]
else:
    rise_time = float("nan")

outside_band = np.where(
    np.abs(actual_speed - setpoint) > 0.02 * setpoint
)[0]

if len(outside_band) == 0:
    settling_time = 0.0
elif outside_band[-1] < len(time) - 1:
    settling_time = time[outside_band[-1] + 1]
else:
    settling_time = float("nan")

disturbance_index = np.searchsorted(time, disturbance_time)
pre_speed = actual_speed[disturbance_index - 1]
post_speed = actual_speed[disturbance_index:]

post_deviation = np.abs(post_speed - setpoint)
max_post_deviation = np.max(post_deviation)

# Recovery time: first point after disturbance that stays
# within 2% of target for the remainder of the simulation.
recovery_time = float("nan")
tolerance = 0.02 * setpoint

for i in range(disturbance_index, len(time)):
    if np.all(np.abs(actual_speed[i:] - setpoint) <= tolerance):
        recovery_time = time[i] - disturbance_time
        break

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, actual_speed, label="Actual Speed")
ax.plot(time, noisy_speed, alpha=0.35, label="Noisy Speed")
ax.plot(time, filtered_speed, label="Filtered Speed")
ax.axhline(setpoint, linestyle="--", label="Desired Speed")
ax.axvline(
    disturbance_time,
    color="red",
    linestyle=":",
    label="Load disturbance"
)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Speed (RPM)")
ax.set_title("Closed-Loop DC Motor Control")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig("motor_response.png", dpi=150)
plt.show()
plt.close(fig)

print("\nCONTROL PERFORMANCE")
print("-------------------")
print(f"Final absolute error: {final_error:.2f} RPM")
print(f"Overshoot: {overshoot:.2f}%")
print(f"Rise time: {rise_time:.2f} seconds")
print(f"Settling time: {settling_time:.2f} seconds")
print(f"Mean absolute error: {np.mean(np.abs(error)):.2f} RPM")
print(f"IAE: {np.sum(np.abs(error)) * dt:.2f}")

print("\nDISTURBANCE ANALYSIS")
print("--------------------")
print(f"Speed before disturbance: {pre_speed:.2f} RPM")
print(f"Maximum deviation from target after disturbance: {max_post_deviation:.2f} RPM")
print(f"Recovery time after disturbance: {recovery_time:.2f} seconds")

print("\nSIMULATION SUMMARY")
print("------------------")
print(f"Target speed: {setpoint} RPM")
print(f"Final simulated speed: {actual_speed[-1]:.2f} RPM")
print(f"Final control voltage: {voltage_values[-1]:.2f} V")
print(f"Maximum voltage: {np.max(voltage_values):.2f} V")
print("Graph saved as motor_response.png")
