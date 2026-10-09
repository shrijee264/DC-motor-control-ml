
class DCMotor:
    def __init__(self):
        self.speed = 0.0
        self.max_speed = 1500.0
        self.time_constant = 0.5

    def update(self, voltage, dt):
        voltage = max(0.0, min(12.0, voltage))

        target_speed = (voltage / 12.0) * self.max_speed

        change = (target_speed - self.speed) / self.time_constant

        self.speed += change * dt

        return self.speed
