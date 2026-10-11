
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from motor import DCMotor
from pid import PIDController


def run_experiment(controller_type):
    dt = 0.01
    duration = 12
    setpoint = 1000
    disturbance_time = 6.0
    load_torque_value = 0.002

    time = np.arange(0, duration, dt)

    motor = DCMotor()

    if controller_type == "P":
        controller = PIDController(
            kp=0.008, ki=0.0, kd=0.0
        )
    elif controller_type == "PI":
        controller = PIDController(
            kp=0.008, ki=0.04, kd=0.0
        )
    else:
        controller = PIDController(
            kp=0.008, ki=0.04, kd=0.0001
        )

    speeds = []
    voltages = []

    for t in time:
        load_torque = (
            load_torque_value if t >= disturbance_time else 0.0
        )

        voltage = controller.update(
            setpoint, motor.speed, dt
        )

        speed = motor.update(
            voltage, dt, load_torque=load_torque
        )

        speeds.append(speed)
        voltages.append(voltage)

    # Convert recorded values to NumPy arrays
    speed_array = np.array(speeds)
    voltage_array = np.array(voltages)

    # Analyze the 1 second immediately before the disturbance
    disturbance_index = np.searchsorted(
        time, disturbance_time
    )

    pre_disturbance_window = speed_array[
        disturbance_index - 100:disturbance_index
    ]

    pre_disturbance_mean = np.mean(pre_disturbance_window)
    pre_disturbance_std = np.std(pre_disturbance_window)

    print(
        f"{controller_type}: pre-disturbance mean = "
        f"{pre_disturbance_mean:.2f} RPM, "
        f"standard deviation = {pre_disturbance_std:.2f} RPM"
    )

    return time, speed_array, voltage_array, setpoint


results = {}
summary = []

for controller_type in ["P", "PI", "PID"]:
    time, speed, voltage, setpoint = run_experiment(
        controller_type
    )

    results[controller_type] = (speed, voltage)

    dt = time[1] - time[0]
    error = setpoint - speed

    disturbance_time = 6.0
    disturbance_index = np.searchsorted(
        time, disturbance_time
    )

    # Use the average speed during the final second before
    # the disturbance as the baseline operating speed.
    pre_disturbance_window = speed[
        disturbance_index - 100:disturbance_index
    ]
    pre_disturbance_mean = np.mean(pre_disturbance_window)

    post_disturbance_speed = speed[disturbance_index:]
    
    recovery_band = 0.02 * max(
        abs(pre_disturbance_mean), 1.0
    )

    recovery_time = float("nan")
    left_band = False

    for i in range(disturbance_index, len(time)):
        deviation = abs(
            speed[i] - pre_disturbance_mean
        )

        if deviation > recovery_band:
            left_band = True

        elif left_band:
            if np.all(
                np.abs(
                    speed[i:] - pre_disturbance_mean
                ) <= recovery_band
            ):
                recovery_time = time[i] - disturbance_time
                break


    # Maximum change from the pre-disturbance operating point
    disturbance_deviation = np.max(
        np.abs(post_disturbance_speed - pre_disturbance_mean)
    )

    # Integrated absolute error after the disturbance,
    # measured relative to the target speed.
    post_disturbance_iae = (
        np.sum(np.abs(setpoint - post_disturbance_speed)) * dt
    )

    overshoot = max(
        0.0,
        (np.max(speed) - setpoint) / setpoint * 100
    )

    summary.append({
        "Controller": controller_type,
        "Final Speed (RPM)": round(speed[-1], 2),
        "Final Error (RPM)": round(abs(error[-1]), 2),
        "Overshoot (%)": round(overshoot, 2),
        "MAE (RPM)": round(np.mean(np.abs(error)), 2),
        "IAE (RPM.s)": round(
            np.sum(np.abs(error)) * dt, 2
        ),
        "Peak Voltage (V)": round(np.max(voltage), 2),
        "Pre-Disturbance Mean (RPM)": round(
            pre_disturbance_mean, 2
        ),
        "Pre-Disturbance Std (RPM)": round(
            np.std(pre_disturbance_window), 2
        ),
        "Max Disturbance Deviation (RPM)": round(
            disturbance_deviation, 2
        ),
        "Recovery Time (s)": round(recovery_time, 2),
        "Post-Disturbance IAE (RPM.s)": round(
            post_disturbance_iae, 2
        )
    })


# Plot controller responses
fig, ax = plt.subplots(figsize=(10, 5))

for controller_type, (speed, voltage) in results.items():
    ax.plot(time, speed, label=controller_type)

ax.axhline(
    setpoint,
    linestyle="--",
    label="Target"
)

ax.axvline(
    6.0,
    color="red",
    linestyle=":",
    label="Load disturbance"
)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Speed (RPM)")
ax.set_title("P vs PI vs PID Controller Comparison")
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig("controller_comparison.png", dpi=150)
plt.show()
plt.close(fig)


# Display and save results
summary_df = pd.DataFrame(summary)

print("\nCONTROLLER COMPARISON")
print(summary_df.to_string(index=False))

summary_df.to_csv(
    "controller_comparison.csv",
    index=False
)

print(
    "\nSaved controller_comparison.png "
    "and controller_comparison.csv"
)
