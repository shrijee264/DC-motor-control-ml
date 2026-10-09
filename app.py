
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from motor import DCMotor
from pid import PIDController
from signal_processing import add_noise, moving_average
from fault_detection import generate_data

st.set_page_config(page_title="Motor Control Simulator", layout="wide")

st.title("DC Motor Control & ML Fault Detection")
st.caption("Python-based engineering simulation | No hardware required")

st.sidebar.header("Simulation Settings")

setpoint = st.sidebar.slider("Target speed (RPM)", 300, 1400, 1000)
kp = st.sidebar.slider("Kp", 0.001, 0.020, 0.008, 0.001)
ki = st.sidebar.slider("Ki", 0.0, 0.100, 0.040, 0.005)
kd = st.sidebar.slider("Kd", 0.0, 0.002, 0.0001, 0.0001)
noise_level = st.sidebar.slider("Sensor noise (RPM)", 0, 100, 10)

if st.button("Run Simulation", type="primary"):
    dt = 0.01
    time = np.arange(0, 10, dt)

    motor = DCMotor()
    controller = PIDController(kp, ki, kd)

    speeds = []
    voltages = []

    for t in time:
        voltage = controller.update(setpoint, motor.speed, dt)
        speed = motor.update(voltage, dt)

        speeds.append(speed)
        voltages.append(voltage)

    actual = np.array(speeds)

    np.random.seed(42)
    noisy = add_noise(actual, noise_level)
    filtered = moving_average(noisy, 10)

    col1, col2, col3 = st.columns(3)
    col1.metric("Target RPM", f"{setpoint}")
    col2.metric("Final RPM", f"{actual[-1]:.1f}")
    col3.metric("Final Voltage", f"{voltages[-1]:.2f} V")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, actual, label="Actual Speed")
    ax.plot(time, noisy, alpha=0.3, label="Noisy Sensor")
    ax.plot(time, filtered, label="Filtered Signal")
    ax.axhline(setpoint, linestyle="--", label="Target")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (RPM)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("ML Fault Detection")

    X, y = generate_data(3000)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    fault = st.selectbox(
        "Choose a simulated operating condition",
        ["Normal", "High Friction", "Overload", "Sensor Fault"]
    )

    examples = {
        "Normal": [1000, 10, 2.0],
        "High Friction": [800, 20, 3.5],
        "Overload": [700, 25, 4.0],
        "Sensor Fault": [1000, 100, 2.0]
    }

    if st.button("Classify Condition"):
        prediction = model.predict([examples[fault]])[0]
        st.write("Selected scenario:", fault)
        st.success(f"Model prediction: {prediction}")

    st.info(
        "Educational prototype. Motor behavior and fault examples are "
        "simulated; predictions are not validated for real equipment."
    )
else:
    st.write(
        "Adjust the parameters in the sidebar and click **Run Simulation** "
        "to explore motor speed control and simulated sensor noise."
    )
