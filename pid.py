
class PIDController:
    def __init__(self, kp, ki, kd, output_min=0.0, output_max=12.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0.0
        self.previous_error = None

    def update(self, setpoint, measured_speed, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        error = setpoint - measured_speed

        if self.previous_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.previous_error) / dt

        candidate_integral = self.integral + error * dt

        raw_output = (
            self.kp * error
            + self.ki * candidate_integral
            + self.kd * derivative
        )

        saturated_output = max(
            self.output_min,
            min(self.output_max, raw_output)
        )

        if raw_output == saturated_output or (
            raw_output > self.output_max and error < 0
        ) or (
            raw_output < self.output_min and error > 0
        ):
            self.integral = candidate_integral

        self.previous_error = error

        return saturated_output
