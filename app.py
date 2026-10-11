
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from motor import DCMotor
from pid import PIDController
from signal_processing import add_noise, moving_average
from fault_detection import generate_data


st.set_page_config(
    page_title="DC Motor Control Simulator",
    page_icon="⚙️",
    layout="wide"
)


# --------------------------------------------------
# Controller comparison simulation
# --------------------------------------------------

def run_controller_experiment(
    controller_type,
    setpoint,
    disturbance_time,
    load_torque_value
):
    dt = 0.01
    duration = 12.0
    time = np.arange(0, duration, dt)

    gains = {
        "P": (0.008, 0.0, 0.0),
        "PI": (0.008, 0.04, 0.0),
        "PID": (0.008, 0.04, 0.0001)
    }

    kp_value, ki_value, kd_value = gains[controller_type]

    motor = DCMotor()
    controller = PIDController(kp_value, ki_value, kd_value)

    speeds = []
    voltages = []
    currents = []
    loads = []

    for t in time:
        current_load = (
            load_torque_value if t >= disturbance_time else 0.0
        )

        voltage = controller.update(setpoint, motor.speed, dt)
        speed = motor.update(
            voltage,
            dt,
            load_torque=current_load
        )

        speeds.append(speed)
        voltages.append(voltage)
        currents.append(motor.current)
        loads.append(current_load)

    return (
        time,
        np.array(speeds),
        np.array(voltages),
        np.array(currents),
        np.array(loads),
        setpoint,
        disturbance_time
    )


# --------------------------------------------------
# Cached synthetic ML model
# --------------------------------------------------

@st.cache_resource
def train_fault_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("DC Motor Control & ML Fault Detection")
st.caption(
    "Interactive DC motor simulation | PID control | "
    "Signal processing | Synthetic ML fault diagnosis"
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Simulation Settings")

setpoint = st.sidebar.slider(
    "Target speed (RPM)",
    min_value=300,
    max_value=1400,
    value=1000
)

kp = st.sidebar.slider(
    "Kp",
    min_value=0.001,
    max_value=0.020,
    value=0.008,
    step=0.001
)

ki = st.sidebar.slider(
    "Ki",
    min_value=0.0,
    max_value=0.100,
    value=0.040,
    step=0.005
)

kd = st.sidebar.slider(
    "Kd",
    min_value=0.0,
    max_value=0.002,
    value=0.0001,
    step=0.0001,
    format="%.4f"
)

noise_level = st.sidebar.slider(
    "Sensor noise (RPM)",
    min_value=0,
    max_value=100,
    value=10
)

st.sidebar.subheader("Load Disturbance")

load_torque = st.sidebar.slider(
    "Additional load torque (N·m)",
    min_value=0.0,
    max_value=0.05,
    value=0.01,
    step=0.001,
    format="%.3f"
)

disturbance_time = st.sidebar.slider(
    "Disturbance starts (seconds)",
    min_value=1.0,
    max_value=9.0,
    value=5.0,
    step=0.5
)

st.sidebar.caption(
    f"Selected load: {load_torque:.3f} N·m"
)


# --------------------------------------------------
# Main motor simulation
# --------------------------------------------------

st.header("Motor Simulation")

if st.button("Run Simulation", type="primary"):
    dt = 0.01
    time = np.arange(0, 10, dt)

    motor = DCMotor()
    controller = PIDController(kp, ki, kd)

    speeds = []
    voltages = []
    currents = []
    loads = []

    for t in time:
        current_load = (
            load_torque if t >= disturbance_time else 0.0
        )

        voltage = controller.update(setpoint, motor.speed, dt)
        speed = motor.update(
            voltage,
            dt,
            load_torque=current_load
        )

        speeds.append(speed)
        voltages.append(voltage)
        currents.append(motor.current)
        loads.append(current_load)

    actual = np.array(speeds)
    voltages = np.array(voltages)
    currents = np.array(currents)
    loads = np.array(loads)

    np.random.seed(42)
    noisy = add_noise(actual, noise_level)
    filtered = moving_average(noisy, 10)

    steady_state_speed = np.mean(actual[-100:])
    steady_state_error = setpoint - steady_state_speed
    peak_speed = np.max(actual)
    overshoot = max(0.0, peak_speed - setpoint)

    saturation_percentage = (
        np.sum(voltages >= 11.99) / len(voltages)
    ) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Target RPM", f"{setpoint:.0f}")
    col2.metric("Final RPM", f"{actual[-1]:.1f}")
    col3.metric("Final Voltage", f"{voltages[-1]:.2f} V")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Steady-State Error",
        f"{steady_state_error:.1f} RPM"
    )
    col2.metric("Peak Speed", f"{peak_speed:.1f} RPM")
    col3.metric("Overshoot", f"{overshoot:.1f} RPM")

    st.metric(
        "Time at Maximum Voltage",
        f"{saturation_percentage:.1f}%"
    )

    # Speed response
    st.subheader("Motor Speed Response")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time, actual, label="Actual Speed")
    ax.plot(time, noisy, alpha=0.3, label="Noisy Sensor")
    ax.plot(time, filtered, label="Filtered Signal")
    ax.axhline(setpoint, linestyle="--", label="Target")
    ax.axvline(
        disturbance_time,
        color="red",
        linestyle=":",
        label="Disturbance Starts"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (RPM)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Motor current
    st.subheader("Motor Current")
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(time, currents, label="Armature Current")
    ax.axvline(
        disturbance_time,
        color="red",
        linestyle=":",
        label="Disturbance Starts"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Current (A)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Control voltage
    st.subheader("Control Voltage")
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(time, voltages, label="Applied Voltage")
    ax.axhline(12.0, linestyle="--", label="Maximum Voltage")
    ax.axvline(
        disturbance_time,
        color="red",
        linestyle=":",
        label="Disturbance Starts"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (V)")
    ax.set_ylim(-0.5, 12.5)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Diagnostics
    st.subheader("Simulation Diagnostics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Maximum Voltage", f"{np.max(voltages):.2f} V")
    col2.metric("Minimum Voltage", f"{np.min(voltages):.2f} V")
    col3.metric("Final Motor Current", f"{currents[-1]:.3f} A")

    st.write("**Load Torque Verification**")
    st.write(f"Selected load torque: {load_torque:.3f} N·m")
    st.write(f"Final applied load: {loads[-1]:.3f} N·m")

    if np.isclose(load_torque, loads[-1]):
        st.success("Selected and final applied load values match.")
    else:
        st.error("Selected and applied load values do not match.")

    if saturation_percentage > 0:
        st.warning(
            "The controller reaches its maximum voltage during "
            "the simulation, limiting its ability to correct "
            "the speed error."
        )

    results_df = pd.DataFrame({
        "Time (s)": time,
        "Actual Speed (RPM)": actual,
        "Noisy Speed (RPM)": noisy,
        "Filtered Speed (RPM)": filtered,
        "Current (A)": currents,
        "Voltage (V)": voltages,
        "Load Torque (N.m)": loads
    })

    st.download_button(
        "Download Simulation Data (CSV)",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="motor_simulation_results.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Adjust the sidebar settings and click Run Simulation "
        "to generate motor response plots."
    )


# --------------------------------------------------
# ML fault detection
# --------------------------------------------------

st.divider()
st.header("ML Fault Detection")

st.write(
    "Classify simulated operating-condition examples using "
    "a Random Forest trained on synthetic data."
)

X, y = generate_data(3000)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = train_fault_model(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("Synthetic Test Accuracy", f"{accuracy:.1%}")
col2.metric("Test Samples", len(y_test))

with st.expander("View classification report"):
    st.text(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

examples = {
    "Normal": [1000, 10, 2.0],
    "High Friction": [800, 20, 3.5],
    "Overload": [700, 25, 4.0],
    "Sensor Fault": [1000, 100, 2.0]
}

fault = st.selectbox(
    "Choose a simulated operating condition",
    list(examples.keys())
)

if st.button("Classify Condition"):
    prediction = model.predict([examples[fault]])[0]
    st.write("Selected scenario:", fault)
    st.success(f"Model prediction: {prediction}")

st.caption(
    "Educational prototype: training data and fault examples "
    "are synthetic. Test accuracy is not validation against "
    "real motor equipment."
)


# --------------------------------------------------
# P vs PI vs PID comparison
# --------------------------------------------------

st.divider()
st.header("P vs PI vs PID Controller Comparison")

st.write(
    "Compare controller responses using the target speed, load "
    "torque, and disturbance time selected in the sidebar. "
    "Each controller uses fixed gains for a fair comparison."
)

if st.button("Compare Controllers"):
    controller_results = {}
    voltage_results = {}
    comparison_rows = []

    for controller_type in ["P", "PI", "PID"]:
        (
            time_cmp,
            speed,
            voltage,
            current,
            loads_cmp,
            target,
            disturbance_time_cmp
        ) = run_controller_experiment(
            controller_type,
            setpoint,
            disturbance_time,
            load_torque
        )

        controller_results[controller_type] = speed
        voltage_results[controller_type] = voltage

        dt_cmp = time_cmp[1] - time_cmp[0]
        error = target - speed

        disturbance_idx = np.searchsorted(
            time_cmp, disturbance_time_cmp
        )

        pre_window = speed[
            max(0, disturbance_idx - 100):disturbance_idx
        ]

        baseline = (
            float(np.mean(pre_window))
            if len(pre_window) > 0
            else float(speed[0])
        )

        post_speed = speed[disturbance_idx:]
        recovery_band = 0.02 * max(abs(baseline), 1.0)
        recovery_time = np.nan

        # Find the earliest time after the disturbance at which
        # speed stays inside the 2% pre-disturbance band.
        for i in range(disturbance_idx, len(time_cmp)):
            if np.all(
                np.abs(speed[i:] - baseline) <= recovery_band
            ):
                recovery_time = time_cmp[i] - disturbance_time_cmp
                break

        disturbance_deviation = (
            float(np.max(np.abs(post_speed - baseline)))
            if len(post_speed) > 0
            else 0.0
        )

        overshoot_pct = max(
            0.0,
            (float(np.max(speed)) - target) / max(abs(target), 1.0) * 100
        )

        comparison_rows.append({
            "Controller": controller_type,
            "Final Speed (RPM)": round(float(speed[-1]), 2),
            "Final Absolute Error (RPM)": round(float(abs(error[-1])), 2),
            "Overshoot (%)": round(overshoot_pct, 2),
            "MAE (RPM)": round(float(np.mean(np.abs(error))), 2),
            "IAE (RPM.s)": round(
                float(np.sum(np.abs(error)) * dt_cmp), 2
            ),
            "Peak Voltage (V)": round(float(np.max(voltage)), 2),
            "Pre-Disturbance Speed (RPM)": round(baseline, 2),
            "Max Disturbance Deviation (RPM)": round(
                disturbance_deviation, 2
            ),
            "Recovery Status": (
                "Recovered"
                if np.isfinite(recovery_time)
                else "Not recovered"
            ),
            "Recovery Time (s)": (
                round(float(recovery_time), 2)
                if np.isfinite(recovery_time)
                else None
            )
        })

    # Speed comparison graph
    st.subheader("Speed Response Comparison")
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 5))

    for controller_type, speed in controller_results.items():
        ax_cmp.plot(time_cmp, speed, label=controller_type)

    ax_cmp.axhline(
        setpoint,
        linestyle="--",
        label="Target"
    )
    ax_cmp.axvline(
        disturbance_time_cmp,
        color="red",
        linestyle=":",
        label="Load Disturbance"
    )
    ax_cmp.set_xlabel("Time (seconds)")
    ax_cmp.set_ylabel("Speed (RPM)")
    ax_cmp.set_title("P vs PI vs PID Controller Comparison")
    ax_cmp.grid(True)
    ax_cmp.legend()
    fig_cmp.tight_layout()
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    # Voltage comparison graph
    st.subheader("Control Voltage Comparison")
    fig_voltage, ax_voltage = plt.subplots(figsize=(10, 4))

    for controller_type, voltage in voltage_results.items():
        ax_voltage.plot(time_cmp, voltage, label=controller_type)

    ax_voltage.axhline(
        12.0,
        linestyle="--",
        label="Maximum Voltage"
    )
    ax_voltage.axvline(
        disturbance_time_cmp,
        color="red",
        linestyle=":",
        label="Load Disturbance"
    )
    ax_voltage.set_xlabel("Time (seconds)")
    ax_voltage.set_ylabel("Voltage (V)")
    ax_voltage.set_ylim(-0.5, 12.5)
    ax_voltage.set_title("P vs PI vs PID Control Voltage")
    ax_voltage.grid(True)
    ax_voltage.legend()
    fig_voltage.tight_layout()
    st.pyplot(fig_voltage)
    plt.close(fig_voltage)

    # Metrics table and CSV export
    comparison_df = pd.DataFrame(comparison_rows)

    st.subheader("Controller Performance Metrics")
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Comparison Results (CSV)",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="controller_comparison.csv",
        mime="text/csv"
    )

    st.caption(
        "Recovery means returning to and remaining within a 2% band "
        "around the average speed during the second before the "
        "disturbance. 'Not recovered' means this condition was not "
        "met before the simulation ended. This is a simulated "
        "educational model."
    )
