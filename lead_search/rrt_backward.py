import numpy as np
from lead_search.rrt import RRT
from common.configuration import SamplingStrategy
from common.node import Node, State
from lead_search.utility_search import check_feasibility_backward, backward_simulation
from typing import Dict, List


class RRTBackward(RRT):
    def __init__(self, simulation_param: Dict, rrt_param: Dict, lead_vehicle_param: Dict):
        """
        Initialization of a RRT planner for forward search

        :param simulation_param: dictionary with simulation parameters
        :param rrt_param: dictionary with RRT parameters
        :param lead_vehicle_param: dictionary with physical parameters of leading vehicle
        """
        super().__init__(simulation_param, rrt_param, lead_vehicle_param)
        if self.sampling_strategy == SamplingStrategy.LOCAL.value:
            self.sampling_range = rrt_param.get("add_sample_range_local_backward")
        else:
            self.sampling_range = rrt_param.get("add_sample_range_global_backward")

    def plan(self, node_list: List[Node], acc_state_list: List[State], lead_vehicle_param: Dict,
             acc_vehicle_param: Dict, simulation_param: Dict) -> Node:
        """
        Planning function for single lane scenario

        :param node_list: list of RRT-nodes from time step t-1
        :param acc_state_list: list with all acc states from time step t
        :param lead_vehicle_param: physical parameters of the leading vehicle
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param simulation_param: parameters of the simulation environment
        :return: list of new nodes of time step t
        """
        # Initialize local variables
        constraints_fulfilled = False
        a_new, sample_v, sample_s_x, state, closest_node, acc_state = None, None, None, None, None, None

        function_calls = {
            "nearest_node": self.nearest_node_global if self.sampling_strategy == "global"
            else self.nearest_node_local,
            "get_input": self.get_input_global if self.sampling_strategy == "global" else self.get_input_local
        }

        # Get maximum and minimum positions for global reference sampling
        if self.sampling_strategy == "global":
            current_min_position, current_max_position = self.get_max_and_min_position(node_list)
            current_min_velocity, current_max_velocity = self.get_max_and_min_velocity(node_list)
        else:
            current_min_position, current_max_position = self.get_max_and_min_position_delta(node_list)
            current_min_velocity, current_max_velocity = self.get_max_and_min_velocity_delta(node_list)

        # Generate new nodes until maximum number nodes is reached
        while not constraints_fulfilled:
            # Sampling
            sample_s_x, sample_v = self.sample(current_min_position, current_max_position,
                                             current_min_velocity, current_max_velocity)

            # Find nearest node using euclidean distance
            closest_node, node_number = function_calls.get("nearest_node")(node_list, np.array([sample_s_x, sample_v]))
            acc_state = acc_state_list[node_number]

            # calculate vehicle input
            a_new = function_calls.get("get_input")(sample_s_x, sample_v, closest_node, acc_state)

            # check input constraints
            a_new = check_feasibility_backward(-1 * a_new[0], closest_node.lead_state.velocity, self.a_min, self.a_max,
                                               self.dt, self.max_velocity, closest_node.lead_state.acceleration,
                                               self.j_min, self.j_max)

            # backward simulation
            inputs = [0, a_new]

            state = backward_simulation(closest_node.lead_state.x_position, closest_node.lead_state.y_position, 0,
                                        closest_node.lead_state.velocity, 0, inputs, self.dt, self.dt, self._vehicle)
            if state[1][3] < 0 or state[0][0] < state[1][0]:
                continue
            else:
                constraints_fulfilled = True

        # create new node
        lead_state = State(state[-1][0], state[-1][1], state[-1][2], state[-1][3], state[-1][4], 0, a_new,
                           closest_node.lead_state.time_step - 1)

        new_node = Node(closest_node, lead_state, None, lead_vehicle_param, acc_vehicle_param, simulation_param)

        return new_node
