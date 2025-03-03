import numpy as np
from common.utility_fcts import check_feasibility
from lead_search.rrt import RRT
from typing import Dict, List
from common.node import Node
from common.state import State
from common.configuration import SamplingStrategy


class RRTForward(RRT):
    def __init__(self, simulation_param: Dict, rrt_param: Dict, lead_vehicle_param: Dict):
        """
        Initialization of a RRT planner for forward search

        :param simulation_param: dictionary with simulation parameters
        :param rrt_param: dictionary with RRT parameters
        :param lead_vehicle_param: dictionary with physical parameters of leading vehicle
        """
        super().__init__(simulation_param, rrt_param, lead_vehicle_param)
        if self.sampling_strategy == SamplingStrategy.LOCAL.value:
            self.sampling_range = rrt_param.get("add_sample_range_local_forward")
        else:
            self.sampling_range = rrt_param.get("add_sample_range_global_forward")

    @staticmethod
    def is_valid_safe_node(new_node: Node) -> bool:
        """
        Validate if new node is safe

        :param new_node: node which is evaluated
        :return: True or False
        """
        if new_node.delta_s <= 0:
            return False
        else:
            return True

    def plan(self, node_list: List[Node], acc_state_list: List[State], lead_vehicle_param: Dict,
             acc_vehicle_param: Dict, simulation_param: Dict) -> List[Node]:
        """
        Planning function for single lane scenario for forward search

        :param node_list: list of RRT-nodes from time step t-1
        :param acc_state_list: list with all acc states from time step t
        :param lead_vehicle_param: physical parameters of the leading vehicle
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param simulation_param: parameters of the simulation environment
        :return: list of new nodes of time step t
        """
        # Initialize local variables
        new_node_list = []
        function_calls = {
            "nearest_node": self.nearest_node_global if self.sampling_strategy == SamplingStrategy.GLOBAL
            else self.nearest_node_local,
            "get_input": self.get_input_global if self.sampling_strategy == SamplingStrategy.GLOBAL
            else self.get_input_local
        }

        # Get maximum and minimum positions for global reference sampling
        if self.sampling_strategy == SamplingStrategy.GLOBAL:
            current_min_position, current_max_position = self.get_max_and_min_position(node_list)
            current_min_velocity, current_max_velocity = self.get_max_and_min_velocity(node_list)
        else:
            current_min_position, current_max_position = self.get_max_and_min_position_delta(node_list)
            current_min_velocity, current_max_velocity = self.get_max_and_min_velocity_delta(node_list)

        # Generate new nodes until maximum number nodes is reached
        while len(new_node_list) != self.number_nodes_rrt:
            # Sampling
            sample_s_x, sample_v = self.sample(current_min_position, current_max_position,
                                               current_min_velocity, current_max_velocity)

            # Find nearest node using euclidean distance
            closest_node, node_number = function_calls.get("nearest_node")(node_list, np.array([sample_s_x, sample_v]))
            acc_state = acc_state_list[node_number]

            # calculate vehicle input
            a_new = function_calls.get("get_input")(sample_s_x, sample_v, closest_node, acc_state)

            # check input constraints
            a_new = check_feasibility(a_new, closest_node.lead_state.velocity, self.a_min, self.a_max, self.dt,
                                      self.max_velocity, closest_node.lead_state.acceleration, self.j_min, self.j_max)

            # forward simulation
            new_node = self.forward_simulation(0, a_new, closest_node, acc_state, lead_vehicle_param,
                                               acc_vehicle_param, simulation_param)

            # validate new node
            if not self.is_valid_safe_node(new_node):
                continue
            new_node_list.append(new_node)

        # delete childless nodes
        self.dereference_unused_nodes(node_list, new_node_list)

        return new_node_list
