from common.node import Node
from acc.acc_interface import AccFactory
from lead_search.rrt_forward import RRTForward
from lead_search.utility_search import init_nodes_forward
from common.utility_fcts import acc_input_forward
from output.storage import store_results
import numpy as np
from typing import List, Dict, Tuple


def check_unsafe_node_reached(node_list: List[Node]) -> Tuple[List[Node], bool]:
    """
    Search for unsafe nodes and evaluate these nodes

    :param node_list: list of nodes generated in last iteration
    :return: unsafe node or node_list passed in parameters and boolean value as indicator for an unsafe node
    """
    unsafe = False
    unsafe_nodes = [node for node in node_list if node.delta_s <= node.unsafe_distance]
    if len(unsafe_nodes) > 0:
        node_list = [unsafe_nodes[0]]
        print("unsafe node")
        unsafe = True
        return node_list, unsafe
    else:
        return node_list, unsafe


def search(rrt_param: Dict, acc_vehicle_param: Dict, simulation_param: Dict, lead_vehicle_param: Dict,
           acc_param: Dict, acc_param_all_controllers: Dict) -> bool:
    """
    Forward search using RRT

    :param rrt_param: dictionary with RRT parameters
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param lead_vehicle_param: dictionary with physical parameters of the leading vehicle
    :param acc_param: dictionary with parameters of selected ACC controller
    :param acc_param_all_controllers: dictionary with parameter dictionaries for each ACC controller
    :return: boolean value indicating that an unsafe state was found
    """
    # Initialize simulation parameters
    num_iterations = simulation_param.get("num_iterations")
    t_react_acc = int(acc_vehicle_param.get("t_react") / simulation_param.get("dt"))
    collision = False

    # Initialize vehicle planners
    lead_planner = RRTForward(simulation_param, rrt_param, lead_vehicle_param)
    acc_planner = AccFactory.create(simulation_param, acc_vehicle_param, acc_param)

    # Initialize list of nodes
    node_list = init_nodes_forward(rrt_param, acc_vehicle_param, simulation_param, lead_vehicle_param)

    # Start of simulation
    for idx in range(num_iterations):
        if simulation_param.get("verbose_mode"):
            print("Iteration: " + str(idx + 1))
        acc_state_list = []

        # Check if unsafe state is reached
        node_list, unsafe = check_unsafe_node_reached(node_list)
        if unsafe:
            break

        # Iterate at time step t over all nodes from time step t-1 and plan ACC
        for current_node in node_list:
            acc_state_list.append(acc_input_forward(acc_planner, current_node, lead_vehicle_param,
                                                    acc_vehicle_param, t_react_acc, simulation_param.get("dt")))

        # Plan the lead vehicle trajectory
        node_list = lead_planner.plan(node_list, acc_state_list, lead_vehicle_param, acc_vehicle_param,
                                      simulation_param)

    if simulation_param.get("store_results") or collision:
        if len(node_list) == 1:
            collision_node_number = 0
        else:
            collision_node_number = np.random.randint(0, len(node_list))
        store_results(node_list[collision_node_number], simulation_param, acc_vehicle_param, lead_vehicle_param,
                      rrt_param, acc_param_all_controllers)
    return collision
