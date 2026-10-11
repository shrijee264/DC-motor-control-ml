
import numpy as np


class DCMotor:
    def __init__(
        self,
        resistance=1.0,
        inductance=0.5,
        torque_constant=0.01,
        back_emf_constant=0.01,
        inertia=0.001,
        friction=0.001,
        max_voltage=12.0,
    ):
        self.R = resistance
        self.L = inductance
        self.Kt = torque_constant
        self.Ke = back_emf_constant
        self.J = inertia
        self.B = friction
        self.max_voltage = max_voltage

        self.current = 0.0
        self.omega = 0.0
        self.speed = 0.0

    def update(self, voltage, dt, load_torque=0.0):
        if dt <= 0:
            raise ValueError("dt must be positive")

        voltage = np.clip(
            voltage, 0.0, self.max_voltage
        )

        back_emf = self.Ke * self.omega

        current_rate = (
            voltage - self.R * self.current - back_emf
        ) / self.L

        torque = self.Kt * self.current

        speed_rate = (
            torque - self.B * self.omega - load_torque
        ) / self.J

        self.current += current_rate * dt
        self.omega += speed_rate * dt

        self.speed = self.omega * 60.0 / (2.0 * np.pi)

        return self.speed
