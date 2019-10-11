import math
from common.utility_safety import safe_distance, unsafe_distance
from typing import Dict, Union
from common.state import State
from vehicleParameters import VehicleParameters
from common.configuration import VehicleType


class Node:
    def __init__(self, parent: Union['Node', None], lead_state: State, acc_state: Union[State, None],
                 lead_vehicle_param: Dict, acc_vehicle_param: Dict, simulation_param: Dict):
        """
        :param parent: parent node
        :param lead_state: Leading vehicle's state with acceleration, velocity, x/y-position, orientation, steering vel.
        :param acc_state: ACC vehicle's state with acceleration, velocity, x/y-position, orientation, steering vel.
        :param lead_vehicle_param: dictionary with physical parameters of leading vehicle
        :param acc_vehicle_param: dictionary with physical parameters of ACC vehicle
        :param simulation_param: dictionary with simulation parameters
        """
        self._lead_state = lead_state
        self._acc_state = acc_state
        self._parent = parent
        if self.acc_state is not None:
            self._delta_s = self.get_lead_x_position_rear(lead_vehicle_param) - \
                            self.get_acc_x_position_front(acc_vehicle_param)
            self._delta_v = lead_state.velocity - acc_state.velocity
            if self.acc_state.acceleration is not None:
                self._safe_distance = safe_distance(self.acc_state.velocity, self.lead_state.velocity,
                                                    acc_vehicle_param.get("a_min"),
                                                    lead_vehicle_param.get("a_min"),
                                                    acc_vehicle_param.get("t_react"),
                                                    self.acc_state.acceleration, self.lead_state.acceleration,
                                                    acc_vehicle_param.get("j_min"),
                                                    lead_vehicle_param.get("j_min"),
                                                    simulation_param.get("dt"),
                                                    self.get_acc_x_position_front(acc_vehicle_param),
                                                    self.get_lead_x_position_rear(lead_vehicle_param),
                                                    acc_vehicle_param.get("a_max"),
                                                    acc_vehicle_param.get("j_max"))
                self._unsafe_distance = unsafe_distance(self.acc_state.velocity, self.lead_state.velocity,
                                                        acc_vehicle_param.get("a_min"),
                                                        lead_vehicle_param.get("a_min"),
                                                        self.acc_state.acceleration, self.lead_state.acceleration,
                                                        acc_vehicle_param.get("j_min"),
                                                        lead_vehicle_param.get("j_min"),
                                                        simulation_param.get("dt"),
                                                        self.get_acc_x_position_front(acc_vehicle_param),
                                                        self.get_lead_x_position_rear(lead_vehicle_param),
                                                        simulation_param.get("v_col"))
            else:
                self._safe_distance = None
                self._unsafe_distance = None
        else:
            self._delta_s = None
            self._delta_v = None
            self._safe_distance = None
            self._unsafe_distance = None

    @property
    def lead_state(self) -> State:
        return self._lead_state

    @lead_state.setter
    def lead_state(self, state: State):
        self._lead_state = state

    @property
    def acc_state(self) -> State:
        return self._acc_state

    @acc_state.setter
    def acc_state(self, state: State):
        self._acc_state = state

    @property
    def parent(self) -> 'Node':
        return self._parent

    @parent.setter
    def parent(self, node: 'Node'):
        self._parent = node

    @property
    def delta_s(self) -> float:
        return self._delta_s

    @delta_s.setter
    def delta_s(self, value: float):
        self._delta_s = value

    @property
    def delta_v(self) -> float:
        return self._delta_v

    @delta_v.setter
    def delta_v(self, value: float):
        self._delta_v = value

    @property
    def safe_distance(self) -> float:
        return self._safe_distance

    @safe_distance.setter
    def safe_distance(self, value: float):
        self._safe_distance = value

    @property
    def unsafe_distance(self) -> float:
        return self._unsafe_distance

    @unsafe_distance.setter
    def unsafe_distance(self, value: float):
        self._unsafe_distance = value

    def add_acc_state(self, acc_state: State, lead_vehicle_param: Dict, acc_vehicle_param: Dict,
                      simulation_param: Dict):
        """
        Update/Add ACC state

        :param acc_state: ACC state with acceleration, velocity, x/y-position, orientation, steering vel.
        :param lead_vehicle_param: dictionary with physical parameters of leading vehicle
        :param acc_vehicle_param: dictionary with physical parameters of ACC vehicle
        :param simulation_param: dictionary with simulation parameters
        """
        self.acc_state = acc_state
        self.delta_s = \
            self.get_lead_x_position_rear(lead_vehicle_param) - self.get_acc_x_position_front(acc_vehicle_param)
        self.delta_v = self.lead_state.velocity - acc_state.velocity
        self.safe_distance = safe_distance(self.acc_state.velocity, self.lead_state.velocity,
                                           acc_vehicle_param.get("a_min"),
                                           lead_vehicle_param.get("a_min"),
                                           acc_vehicle_param.get("t_react"),
                                           self.acc_state.acceleration, self.lead_state.acceleration,
                                           acc_vehicle_param.get("j_min"),
                                           lead_vehicle_param.get("j_min"),
                                           simulation_param.get("dt"),
                                           self.get_acc_x_position_front(acc_vehicle_param),
                                           self.get_lead_x_position_rear(lead_vehicle_param),
                                           acc_vehicle_param.get("a_max"),
                                           acc_vehicle_param.get("j_max"))
        self.unsafe_distance = unsafe_distance(self.acc_state.velocity, self.lead_state.velocity,
                                               acc_vehicle_param.get("a_min"),
                                               lead_vehicle_param.get("a_min"),
                                               self.acc_state.acceleration, self.lead_state.acceleration,
                                               acc_vehicle_param.get("j_min"),
                                               lead_vehicle_param.get("j_min"),
                                               simulation_param.get("dt"),
                                               self.get_acc_x_position_front(acc_vehicle_param),
                                               self.get_lead_x_position_rear(lead_vehicle_param),
                                               simulation_param.get("v_col"))

    def get_acc_x_position_front(self, acc_vehicle_param: Dict) -> float:
        """
        Calculates the front position of the ACC vehicle

        :param acc_vehicle_param: dictionary with physical parameters of ACC vehicle
        """
        vehicle_length = acc_vehicle_param.get("dynamics_param").l
        return self.acc_state.x_position + vehicle_length / 2

    def get_lead_x_position_rear(self, lead_vehicle_param: Dict) -> float:
        """
        Calculates the rear position of the lead vehicle
        :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
        """
        vehicle_param = lead_vehicle_param.get("dynamics_param")
        return min(self.lead_state.x_position, self.x_position_left_front("lead_vehicle", vehicle_param),
                   self.x_position_right_front("lead_vehicle", vehicle_param),
                   self.x_position_left_rear("lead_vehicle", vehicle_param),
                   self.x_position_right_rear("lead_vehicle", vehicle_param))

    def x_position_left_front(self, vehicle, vehicle_param: VehicleParameters) -> float:
        """
        Calculates the front left position of a given vehicle

        :param vehicle: String for defining the vehicle type
        :param vehicle_param: physical parameters of the vehicle
        """
        if vehicle == VehicleType.LEAD.value:
            return self.lead_state.x_position - ((vehicle_param.w / 2.0) * math.sin(self.lead_state.yaw_angle)) + \
                   ((vehicle_param.l / 2.0) * math.cos(self.lead_state.yaw_angle))
        else:
            return self.acc_state.x_position - \
                   ((vehicle_param.w / 2.0) * math.sin(self.acc_state.yaw_angle)) + \
                   ((vehicle_param.l / 2.0) * math.cos(self.acc_state.yaw_angle))

    def x_position_right_front(self, vehicle, vehicle_param: VehicleParameters) -> float:
        """
        Calculates the front right position of a given vehicle

        :param vehicle: String for defining the vehicle type
        :param vehicle_param: physical parameters of the vehicle
        """
        if vehicle == VehicleType.LEAD.value:
            return self.lead_state.x_position + ((vehicle_param.w / 2.0) * math.sin(self.lead_state.yaw_angle)) + \
                   ((vehicle_param.l / 2.0) * math.cos(self.lead_state.yaw_angle))
        else:
            return self.acc_state.x_position + \
                   ((vehicle_param.w / 2.0) * math.sin(self.acc_state.yaw_angle)) + \
                   ((vehicle_param.l / 2.0) * math.cos(self.acc_state.yaw_angle))

    def x_position_left_rear(self, vehicle, vehicle_param: VehicleParameters) -> float:
        """
        Calculates the rear left position of a given vehicle

        :param vehicle: String for defining the vehicle type
        :param vehicle_param: physical parameters of the vehicle
        """
        if vehicle == VehicleType.LEAD.value:
            return self.lead_state.x_position - ((vehicle_param.w / 2.0) * math.sin(self.lead_state.yaw_angle)) - \
                   ((vehicle_param.l / 2.0) * math.cos(self.lead_state.yaw_angle))
        else:
            return self.acc_state.x_position - \
                   ((vehicle_param.w / 2.0) * math.sin(self.acc_state.yaw_angle)) - \
                   ((vehicle_param.l / 2.0) * math.cos(self.acc_state.yaw_angle))

    def x_position_right_rear(self, vehicle, vehicle_param: VehicleParameters) -> float:
        """
        Calculates the rear right position of a given vehicle

        :param vehicle: String for defining the vehicle type
        :param vehicle_param: physical parameters of the vehicle
        """
        if vehicle == VehicleType.LEAD.value:
            return self.lead_state.x_position + ((vehicle_param.w / 2.0) * math.sin(self.lead_state.yaw_angle)) - \
                   ((vehicle_param.l / 2.0) * math.cos(self.lead_state.yaw_angle))
        else:
            return self.acc_state.x_position + \
                   ((vehicle_param.w / 2.0) * math.sin(self.acc_state.yaw_angle)) - \
                   ((vehicle_param.l / 2.0) * math.cos(self.acc_state.yaw_angle))
