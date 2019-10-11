"""
Abstract RRT class with all functions for the different RRT sub-types.
"""
import sys
from common.node import Node
from common.utility_fcts import forward_simulation, check_feasibility
from common.state import State
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from common.configuration import SamplingStrategy


class RRT(ABC):
    def __init__(self, simulation_param: Dict, rrt_param: Dict, lead_vehicle_param: Dict):
        """
        Initialization of the RRT planner

        :param simulation_param: dictionary with parameters of the simulation environment
        :param rrt_param: dictionary with RRT parameters
        :param lead_vehicle_param: parameters of the leading vehicle
        """
        self._dt = simulation_param.get("dt")
        self._a_max = lead_vehicle_param.get("a_max")
        self._a_min = lead_vehicle_param.get("a_min")
        self._j_max = lead_vehicle_param.get("j_max")
        self._j_min = lead_vehicle_param.get("j_min")
        self._max_velocity = lead_vehicle_param.get("dynamics_param").longitudinal.v_max
        self._t_react = lead_vehicle_param.get("t_react")
        self._vehicle = lead_vehicle_param.get("dynamics_param")
        self._sampling_strategy = rrt_param.get("sampling_strategy")
        self._number_nodes_rrt = rrt_param.get("number_nodes")
        self._sampling_range = None

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        return self._sampling_strategy

    @sampling_strategy.setter
    def sampling_strategy(self, strategy: SamplingStrategy):
        self._sampling_strategy = strategy

    @property
    def a_max(self) -> float:
        return self._a_max

    @a_max.setter
    def a_max(self, value: float):
        self._a_max = value

    @property
    def a_min(self) -> float:
        return self._a_min

    @a_min.setter
    def a_min(self, value: float):
        self._a_min = value

    @property
    def j_max(self) -> float:
        return self._j_max

    @j_max.setter
    def j_max(self, value: float):
        self._j_max = value

    @property
    def j_min(self) -> float:
        return self._j_min

    @j_min.setter
    def j_min(self, value: float):
        self._j_min = value

    @property
    def max_velocity(self) -> float:
        return self._max_velocity

    @max_velocity.setter
    def max_velocity(self, value: float):
        self._max_velocity = value

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value: float):
        self._dt = value

    @property
    def number_nodes_rrt(self) -> int:
        return self._number_nodes_rrt

    @number_nodes_rrt.setter
    def number_nodes_rrt(self, num_nodes: int):
        self._number_nodes_rrt = num_nodes

    @property
    def sampling_range(self) -> List[float]:
        return self._sampling_range

    @sampling_range.setter
    def sampling_range(self, values: List[float]):
        self._sampling_range = values

    def sample(self, current_min_position: float, current_max_position: float, current_min_velocity: float,
               current_max_velocity: float) -> Tuple[float, float]:
        """
        Sampling of desired position/velocity (difference)

        :param current_min_position: current minimum position of all rrt-nodes from time step t-1 [m]
        :param current_max_position: current maximum position of all rrt-nodes from time step t-1 [m]
        :param current_min_velocity: current minimum velocity of all rrt-nodes from time step t-1 [m/s]
        :param current_max_velocity: current maximum velocity of all rrt-nodes from time step t-1 [m/s]
        :return: Sampled x-position (difference) [m] and velocity (difference) [m/s]
        """
        s_x_desired = np.random.uniform(current_min_position - self.sampling_range[0],
                                        current_max_position + self.sampling_range[1])
        v_desired = np.random.uniform(current_min_velocity - self.sampling_range[2],
                                      current_max_velocity + self.sampling_range[3])
        return s_x_desired, v_desired

    @staticmethod
    def normalize_state(list_s: List[float], list_v: List[float], sample: List[float]) \
            -> Tuple[float, float, float, float, float, float]:
        """
        Creates normalized states for global (local) coordinate system

        :param list_s: list of absolute position (difference) values from all node at t-1 [m]
        :param list_v: list of absolute velocity (difference) values from all node at t-1 [m]
        :param sample: list containing the sampled x-position (distance) and velocity (difference)
        :return: mean position (difference) , mean velocity (difference) , position (difference) variance,
        velocity (difference) variance, normalized position (difference) , normalized velocity (difference)
        """
        mean_s = float(np.mean(list_s))
        mean_v = float(np.mean(list_v))
        variance_s = float(np.var(list_s))
        variance_v = float(np.var(list_v))
        if variance_v <= 1e-12:  # necessary to prevent numerical errors
            variance_v = 1.0
        if variance_s <= 1e-12:  # necessary to prevent numerical errors
            variance_s = 1.0
        normalized_s = (sample[0] - mean_s) / variance_s
        normalized_v = (sample[1] - mean_v) / variance_v

        return mean_s, mean_v, variance_s, variance_v, normalized_s, normalized_v

    @staticmethod
    def nearest_node_local(node_list: List[Node], sample: List[float]) -> Tuple[Node, int]:
        """
        Closest node based on Euclidean distance for local coordinate system

        :param node_list: list of nodes from previous time step
        :param sample: list with sampled distance [m], sampled velocity difference [m/s]
        :return: parent node, node index in node_list
        """
        distance_min = sys.maxsize
        node_min = None
        node_number = 0

        # Calculate means and variances for normalization of distance and velocity difference
        # based on all nodes from previous time step and sample
        list_delta_s = [sample[0]]
        list_delta_v = [sample[1]]
        for index, node in enumerate(node_list):
            list_delta_s.append(node.delta_s)
            list_delta_v.append(node.delta_v)
        mean_delta_s, mean_delta_v, variance_delta_s, variance_delta_v, normalized_position, normalized_velocity = \
            RRT.normalize_state(list_delta_s, list_delta_v, sample)

        # Evaluate distance to all nodes and store the one with minimal distance
        for index, node in enumerate(node_list):
            distance = math.sqrt((((node.delta_s - mean_delta_s) / variance_delta_s) - normalized_position) ** 2 +
                                 (((node.delta_v - mean_delta_v) / variance_delta_v) - normalized_velocity) ** 2)
            if distance < distance_min:
                distance_min = distance
                node_min = node
                node_number = index
        return node_min, node_number

    @staticmethod
    def nearest_node_global(node_list: List[Node], sample: List[float]) -> Tuple[Node, int]:
        """
        Euclidean distance to choose the closest node for the global coordinate system

        :param node_list: list of nodes from previous time step
        :param sample:  list with sampled x-position [m], sampled velocity [m/s]
        :return: parent node, node index in node_list
        """
        distance_min = sys.maxsize
        node_number = 0
        node_min = None

        # Calculate means and variances for normalization of position and velocity
        # based on all nodes from previous time step and sample
        list_s = [sample[0]]
        list_v = [sample[1]]
        for index, node in enumerate(node_list):
            list_s.append(node.lead_state.x_position)
            list_v.append(node.lead_state.velocity)

        mean_s, mean_v, variance_s, variance_v, normalized_position, normalized_velocity = \
            RRT.normalize_state(list_s, list_v, sample)

        # Evaluate distance to all nodes and store the one with minimal distance
        for index, node in enumerate(node_list):
            distance = math.sqrt((((node.lead_state.x_position - mean_s) / variance_s) - normalized_position) ** 2 +
                                 (((node.lead_state.velocity - mean_v) / variance_v) - normalized_velocity) ** 2)
            if distance < distance_min:
                distance_min = distance
                node_min = node
                node_number = index
        return node_min, node_number

    def get_input_local(self, delta_s_desired: float, delta_v_desired: float, closest_node: Node, acc_state: State) \
            -> float:
        """
        Calculates the input to get to the sampled state for the coordinate system

        :param delta_s_desired: sampled x-position for the leading vehicle [m]
        :param delta_v_desired:  sampled velocity for the leading vehicle [m/s]
        :param closest_node: nearest node to sampled state
        :param acc_state: ACC state at time step t
        :return: input acceleration
        """
        x_lead, v_lead = closest_node.lead_state.x_position, closest_node.lead_state.velocity

        a = np.transpose(np.linalg.pinv(np.array([[0.5 * self._dt ** 2, self._dt]])))
        desired = np.array([[delta_s_desired], [delta_v_desired]])
        new_state_following = np.array([[acc_state.x_position], [acc_state.velocity]])
        time = np.array([[1, self._dt], [0, 1]])
        lead = np.array([[x_lead], [v_lead]])
        tmp_1 = np.add(desired, new_state_following)
        tmp_2 = np.dot(time, lead)
        tmp_3 = np.subtract(tmp_1, tmp_2)
        inputs = np.dot(a, tmp_3)

        return inputs[0]

    def get_input_global(self, s_desired: float, v_desired: float, closest_node: Node, acc_state: State) -> float:
        """
        Calculates the input to get to the sampled state for the global references sampling strategy

        :param s_desired: sampled position for the leading vehicle [m]
        :param v_desired:  sampled velocity for the leading vehicle [m/s]
        :param closest_node: nearest node to sampled state
        :param acc_state: ACC state at time step t - not used for input calculation (not needed for global
        coordinate system)
        :return: input acceleration
        """
        current_position, current_velocity = closest_node.lead_state.x_position, closest_node.lead_state.velocity

        a = np.transpose(np.linalg.pinv(np.array([[0.5 * self._dt ** 2, self._dt]])))
        desired = np.array([[s_desired], [v_desired]])
        time = np.array([[1, self._dt], [0, 1]])
        current_state = np.array([[current_position], [current_velocity]])
        tmp = np.subtract(desired, np.dot(time, current_state))
        inputs = np.dot(a, tmp)

        return inputs[0]

    def braking(self, closest_node: Node, acc_state: State, lead_vehicle_param: Dict, acc_vehicle_param: Dict,
                simulation_param: Dict) -> Node:
        """
        Braking with minimal acceleration until standstill for a single node which violates safe distance

        :param closest_node: node which violates safe distance at time step t-1
        :param acc_state: corresponding ACC states at time step t
        :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
        :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
        :param simulation_param: dictionary with parameters of the simulation environment
        :return: single node at time step t
        """
        a_new = self.a_min
        a_new = check_feasibility(a_new, closest_node.lead_state.velocity, self.a_min, self.a_max, self.dt,
                                  self.max_velocity, closest_node.lead_state.acceleration, self.j_min, self.j_max)

        # forward simulation
        inputs = [0, a_new]
        state = forward_simulation(closest_node.lead_state.x_position, closest_node.lead_state.y_position, 0,
                                   closest_node.lead_state.velocity, 0, inputs, self.dt, self.dt, self._vehicle)

        # create new node
        lead_state = State(state[-1][0], state[-1][1], state[-1][2], state[-1][3], state[-1][4], 0, float(a_new))

        new_node = Node(closest_node, lead_state, acc_state, lead_vehicle_param, acc_vehicle_param, simulation_param)
        return new_node

    def forward_simulation(self, steering_velocity: float, acceleration: float, closest_node: Node, acc_state: State,
                           lead_vehicle_param: Dict, acc_vehicle_param: Dict, simulation_param: Dict) -> Node:
        """
        Forward simulation of a node

        :param steering_velocity: input steering velocity [rad/s]
        :param acceleration: input acceleration [m/s^2]
        :param closest_node: node which violates safe distance at time step t-1
        :param acc_state: corresponding ACC states at time step t
        :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
        :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
        :param simulation_param: dictionary with parameters of the simulation environment
        :return: single node at time step t
        """
        inputs = [steering_velocity, acceleration]
        state = forward_simulation(closest_node.lead_state.x_position, closest_node.lead_state.y_position,
                                   closest_node.lead_state.steering_angle, closest_node.lead_state.velocity, 0,
                                   inputs, self.dt, self.dt, self._vehicle)

        # create new node
        lead_state = State(state[-1][0], state[-1][1], state[-1][2], state[-1][3], state[-1][4],
                           steering_velocity, float(acceleration), closest_node.lead_state.time_step + 1)
        new_node = Node(closest_node, lead_state, acc_state, lead_vehicle_param, acc_vehicle_param, simulation_param)

        return new_node

    @staticmethod
    def dereference_unused_nodes(node_list: List[Node], new_node_list: List[Node]):
        """
        Delete all nodes from time step t-1 which are childless

        :param node_list: list of nodes from time step t-1
        :param new_node_list: list of nodes from time step t
        """
        parent_status = False
        for current_parent_node in node_list:
            for current_child_node in new_node_list:
                if current_child_node.parent == current_parent_node:
                    parent_status = True
                    break
            if not parent_status:
                current_parent_node.parent = None
                node_list.remove(current_parent_node)

    @staticmethod
    def get_max_and_min_position(node_list: List[Node]) -> Tuple[float, float]:
        """
        Get maximum and minimum position of time step t-1

        :param node_list: list of nodes from time step t-1
        :return: minimum and maximum position [m]
        """
        current_min_position = sys.maxsize
        current_max_position = 0

        for node in node_list:
            if node.lead_state.x_position > current_max_position:
                current_max_position = node.lead_state.x_position
            if node.lead_state.x_position < current_min_position:
                current_min_position = node.lead_state.x_position

        return current_min_position, current_max_position

    @staticmethod
    def get_max_and_min_velocity(node_list: List[Node]) -> Tuple[float, float]:
        """
        Get maximum and minimum position of time step t-1

        :param node_list: list of nodes from time step t-1
        :return: minimum and maximum velocity [m/s]
        """
        current_max_v = 0
        current_min_v = sys.maxsize

        for node in node_list:
            if node.lead_state.velocity > current_max_v:
                current_max_v = node.lead_state.velocity
            if node.lead_state.velocity < current_min_v:
                current_min_v = node.lead_state.velocity

        return current_min_v, current_max_v

    @staticmethod
    def get_max_and_min_position_delta(node_list: List[Node]) -> Tuple[float, float]:
        """
        Get maximum and minimum position of time step t-1

        :param node_list: list of nodes from time step t-1
        :return: minimum and maximum position difference [m]
        """
        current_min_position_delta = sys.maxsize
        current_max_position_delta = 0

        for node in node_list:
            if node.delta_s > current_max_position_delta:
                current_max_position_delta = node.delta_s
            if node.delta_s < current_min_position_delta:
                current_min_position_delta = node.delta_s

        return current_min_position_delta, current_max_position_delta

    @staticmethod
    def get_max_and_min_velocity_delta(node_list: List[Node]) -> Tuple[float, float]:
        """
        Get maximum and minimum position of time step t-1

        :param node_list: list of nodes from time step t-1
        :return: minimum and maximum velocity difference [m/s]
        """
        current_max_velocity_delta = -sys.maxsize + 1
        current_min_velocity_delta = sys.maxsize

        for node in node_list:
            if node.delta_v > current_max_velocity_delta:
                current_max_velocity_delta = node.delta_v
            if node.delta_v < current_min_velocity_delta:
                current_min_velocity_delta = node.delta_v

        return current_min_velocity_delta, current_max_velocity_delta

    @abstractmethod
    def plan(self, node_list: List[Node], acc_state_list: List[State], lead_vehicle_param: Dict,
             acc_vehicle_param: Dict, simulation_param: Dict) -> List[Node]:
        """
        Planning function for single lane scenario

        :param node_list: list of rrt-nodes from time step t-1
        :param acc_state_list: list with all acc states from time step t
        :param lead_vehicle_param: physical parameters of the vehicle
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param simulation_param: parameters of the simulation environment
        :return: list of new nodes of time step t
        """
        pass
