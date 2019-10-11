class State:
    def __init__(self, x_position: float, y_position: float, steering_angle: float, velocity: float, yaw_angle: float,
                 steering_velocity: float, acceleration: float, time_step: int):
        """
        Initialization of a vehicle state

        :param x_position: x-position of vehicle [m]
        :param y_position: y-position of vehicle [m]
        :param steering_angle: steering angle of vehicle [rad]
        :param velocity: velocity of vehicle [m/s]
        :param yaw_angle: yaw angle of vehicle [rad]
        :param steering_velocity: steering velocity of vehicle [rad/s]
        :param acceleration: acceleration of vehicle [m/s^2]
        """
        self._x_position = x_position
        self._y_position = y_position
        self._steering_angle = steering_angle
        self._velocity = velocity
        if abs(self._velocity) < 1e-8:
            self._velocity = 0
        self._yaw_angle = yaw_angle
        self._steering_velocity = steering_velocity
        self._acceleration = acceleration
        self._time_step = time_step

    @property
    def x_position(self) -> float:
        return self._x_position

    @x_position.setter
    def x_position(self, value: float):
        self._x_position = value

    @property
    def y_position(self) -> float:
        return self._y_position

    @y_position.setter
    def y_position(self, value: float):
        self._y_position = value

    @property
    def steering_angle(self) -> float:
        return self._steering_angle

    @steering_angle.setter
    def steering_angle(self, value: float):
        self._steering_angle = value

    @property
    def velocity(self) -> float:
        return self._velocity

    @velocity.setter
    def velocity(self, value: float):
        self._velocity = value

    @property
    def yaw_angle(self) -> float:
        return self._yaw_angle

    @yaw_angle.setter
    def yaw_angle(self, value: float):
        self._yaw_angle = value

    @property
    def steering_velocity(self) -> float:
        return self._steering_velocity

    @steering_velocity.setter
    def steering_velocity(self, value: float):
        self._steering_velocity = value

    @property
    def acceleration(self) -> float:
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value: float):
        self._acceleration = value

    @property
    def time_step(self) -> int:
        return self._time_step

    @time_step.setter
    def time_step(self, value: float):
        self._time_step = value
