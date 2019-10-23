from acc.acc_interface import AccFactory
from output.storage import store_results
from lead_search.utility_search import init_nodes_forward
from typing import Dict
import numpy as np
from common.utility_fcts import check_feasibility, acc_input_forward
from lead_search.rrt import RRT


class MonteCarlo(RRT):
    def __init__(self, simulation_param, rrt_param, lead_vehicle_param):
        """
        Initialization of a Monte Carlo planner for forward search

        :param simulation_param: dictionary with simulation parameters
        :param rrt_param: dictionary with RRT parameters
        :param lead_vehicle_param: dictionary with physical parameters of lead vehicle
        """
        super().__init__(simulation_param, rrt_param, lead_vehicle_param)
        self._beta_a = rrt_param.get("mcs_beta_a")
        self._beta_b = rrt_param.get("mcs_beta_b")

    # ----------------------------------------------Planning functions--------------------------------------------------
    def plan(self, node_list, acc_state_list, lead_vehicle_param, acc_vehicle_param, simulation_param):
        """
        Planning function for single lane scenario

        :param node_list: list of rrt-nodes from time step t-1
        :param acc_state_list: list with all acc states from time step t
        :param lead_vehicle_param: physical parameters of the lead vehicle
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param simulation_param: parameters of the simulation environment
        :return: list of new nodes of time step t
        """
        new_node_list = []
        for i in range(len(node_list)):
            current_node = node_list[i]
            # Sampling
            a_new = \
                self.a_min + np.random.beta(self._beta_a, self._beta_b) * \
                (self.a_max - self.a_min)

            # check input constraints
            a_new = check_feasibility(a_new, current_node.lead_state.velocity, self.a_min,
                                      self.a_max, self.dt, self.max_velocity,
                                      current_node.lead_state.acceleration, self.j_min, self.j_max)

            # forward simulation
            acc_state = acc_state_list[i]
            new_node = self.forward_simulation(0, a_new, current_node, acc_state, lead_vehicle_param,
                                               acc_vehicle_param, simulation_param)
            new_node_list.append(new_node)

        return new_node_list


def search(rrt_param: Dict, acc_vehicle_param: Dict, simulation_param: Dict, lead_vehicle_param, acc_param: Dict,
           acc_param_all_controllers: Dict):
    """
    Monte Carlo simulation

    :param rrt_param: dictionary with RRT parameters
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param acc_param: dictionary with parameters of selected ACC controller
    :param acc_param_all_controllers: dictionary with parameter dictionaries for each ACC controller
    """
    # Initialize vehicle planners
    lead_planner = MonteCarlo(simulation_param, rrt_param, lead_vehicle_param)
    acc_planner = AccFactory.create(simulation_param, acc_vehicle_param, acc_param)

    # Initialization of variables for simulation
    collision = False
    num_iterations = simulation_param.get("num_iterations")
    t_react_acc = int(acc_vehicle_param.get("t_react") / simulation_param.get("dt"))
    node_list = init_nodes_forward(rrt_param, acc_vehicle_param, simulation_param, lead_vehicle_param)

    # Start of simulation
    for idx in range(num_iterations):
        if simulation_param.get("verbose_mode"):
            print("Iteration: " + str(idx + 1))
        acc_state_list = []

        # Iterate at time step t over all nodes from time step t-1 and plan ACC
        for current_node in node_list:
            acc_state_list.append(acc_input_forward(acc_planner, current_node, lead_vehicle_param, acc_vehicle_param,
                                                    t_react_acc, simulation_param.get("dt")))

        # Plan the lead vehicle trajectory
        node_list = lead_planner.plan(node_list, acc_state_list, lead_vehicle_param, acc_vehicle_param,
                                      simulation_param)

        #  Evaluation if collision occurred
        for index, node in enumerate(node_list):
            if node.delta_s <= 0:
                collision = True
                node_list = [node]
                print("collision")
                break
        if collision:
            break

    # Store results
    if simulation_param.get("safe_results"):
        if len(node_list) == 1:
            output_node_number = 0
        else:
            output_node_number = np.random.randint(0, len(node_list))
        store_results(node_list[output_node_number], simulation_param, acc_vehicle_param, lead_vehicle_param,
                      rrt_param, acc_param_all_controllers)
