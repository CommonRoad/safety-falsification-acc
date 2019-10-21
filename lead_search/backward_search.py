from acc.acc_interface import AccFactory
from lead_search.rrt_backward import RRTBackward
from lead_search.rrt_forward import RRTForward
from common.state import State
from common.node import Node
from common.utility_fcts import check_feasibility, forward_propagation, acc_input_forward
from output.storage import store_results
import numpy as np
from timeit import default_timer as timer
from lead_search.utility_search import check_feasibility_backward
from typing import Dict, List, Tuple


def check_unsafe_node_reached(node: Node, simulation_param: Dict, rrt_param: Dict, lead_vehicle_param: Dict,
                              acc_vehicle_param: Dict, acc_planner: AccFactory) -> Tuple[bool, bool]:
    """
    Evaluate if unsafe node is reached after forward simulation and if goal node is valid

    :param node: starting node for forward simulation
    :param simulation_param: dictionary with parameters of the simulation environment
    :param rrt_param: rrt_param: dictionary with RRT parameters
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    :param acc_planner: ACC planning algorithm
    :return: valid unsafe node, initial safe node
    """
    dt = simulation_param.get("dt")
    t_react_acc = int(acc_vehicle_param.get("t_react") / dt)
    current_node = node
    lead_planner = RRTForward(simulation_param, rrt_param, lead_vehicle_param)
    lead_profile = []

    # Check if initial node is safe
    if node.safe_distance < node.delta_s:
        initial_node_safe = True
    else:
        initial_node_safe = False

    while current_node.parent:
        lead_profile.append(current_node.lead_state.acceleration)
        current_node = current_node.parent

    for i in range(len(lead_profile)):
        # Plan the ACC trajectory based on the selected controller
        acc_state = acc_input_forward(acc_planner, node, lead_vehicle_param, acc_vehicle_param, t_react_acc, dt)

        # Forward simulation of lead vehicle
        a_new = check_feasibility(lead_profile[i], node.lead_state.velocity,
                                  lead_vehicle_param.get("a_min"), lead_vehicle_param.get("a_max"),
                                  simulation_param.get("dt"),
                                  lead_vehicle_param.get("dynamics_param").longitudinal.v_max,
                                  node.lead_state.acceleration, lead_vehicle_param.get("j_min"),
                                  lead_vehicle_param.get("j_max"))

        node = lead_planner.forward_simulation(0, a_new, node, acc_state, lead_vehicle_param,
                                               acc_vehicle_param, simulation_param)

    # Check if safe distance is violated and if collision is valid in case initial node was safe
    if node.delta_s <= node.unsafe_distance:
        return True, initial_node_safe
    else:
        return False, initial_node_safe


def init_nodes_backward(acc_vehicle_param: Dict, lead_vehicle_param: Dict, simulation_param: Dict,
                        rrt_param: Dict) -> List[Node]:
    """
    Evaluate if unsafe node is reached after forward simulation and if goal node is valid

    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param rrt_param: dictionary with RRT parameters
    :return: list of new nodes
    """
    new_node_list = []
    while len(new_node_list) < rrt_param.get("number_nodes"):
        v_lead = np.random.uniform(0, lead_vehicle_param.get("dynamics_param").longitudinal.v_max)
        v_acc = np.random.uniform(0, acc_vehicle_param.get("dynamics_param").longitudinal.v_max)
        a_lead = np.random.uniform(lead_vehicle_param.get("a_min"), lead_vehicle_param.get("a_max"))
        a_acc = np.random.uniform(acc_vehicle_param.get("a_min"),  acc_vehicle_param.get("a_max"))
        s_x_lead = np.random.uniform(acc_vehicle_param.get("dynamics_param").l / 2 +
                                     lead_vehicle_param.get("dynamics_param").l / 2,
                                     acc_vehicle_param.get("dynamics_param").l / 2 +
                                     lead_vehicle_param.get("dynamics_param").l / 2 +
                                     simulation_param.get("delta_s_init_max"))
        acc_state = State(0, 1.75, 0, v_acc, 0, 0, a_acc, 0)
        lead_state = State(s_x_lead, 1.75, 0, v_lead, 0, 0, a_lead, 0)

        new_node = Node(None, lead_state, acc_state, lead_vehicle_param, acc_vehicle_param, simulation_param)
        if 0 <= new_node.unsafe_distance - new_node.delta_s <= simulation_param.get("s_safe_init_backward"):
            new_node_list.append(new_node)
        else:
            continue

    return new_node_list


def acc_random_backward(node_list: List[Node], acc_vehicle_param: Dict, dt: float) -> List[State]:
    """
    Initialize list of nodes for forward search

    :param node_list: list of nodes at time step t
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    :param dt: time step size [s]
    :return: list of acc states at time step t-1
    """
    acc_state_list = []
    for i, node in enumerate(node_list):
        s_x_acc, v_acc, a_acc = node.acc_state.x_position, node.acc_state.velocity, \
                                np.random.uniform(node.acc_state.acceleration - acc_vehicle_param.get("j_max"),
                                                  node.acc_state.acceleration - acc_vehicle_param.get("j_min"))
        a_acc = check_feasibility_backward(-1 * a_acc, v_acc, acc_vehicle_param.get("a_min"),
                                           acc_vehicle_param.get("a_max"), dt,
                                           acc_vehicle_param.get("dynamics_param").longitudinal.v_max,
                                           node.acc_state.acceleration, acc_vehicle_param.get("j_min"),
                                           acc_vehicle_param.get("j_max"))

        # state of the following vehicle at time step t - 1
        new_acc_position = s_x_acc - (0.5 * a_acc * dt ** 2) - (v_acc * dt)
        new_acc_velocity = v_acc + a_acc * dt
        acc_state = State(new_acc_position, node.acc_state.y_position, node.acc_state.steering_angle,
                          new_acc_velocity, node.acc_state.yaw_angle, 0, a_acc, node.acc_state.time_step + 1)
        acc_state_list.append(acc_state)

    return acc_state_list


def is_valid_initial_node(node: Node) -> bool:
    """
    Validate if current node is a valid solution

    :param node: node to evaluate
    return boolean indicating if node is a valid initial node
    """
    safe_distance = False
    collision = True

    if node.delta_s > node.safe_distance:
        safe_distance = True
    if node.delta_s > 0:
        collision = False
    if safe_distance and not collision:
        return True
    else:
        return False


def search(rrt_param: Dict, acc_vehicle_param: Dict, simulation_param: Dict, lead_vehicle_param: Dict,
           acc_param: Dict, acc_param_all_controllers: Dict):
    """
    Backward search using RRT

    :param rrt_param: dictionary with RRT parameters
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param lead_vehicle_param: dictionary with physical parameters of the leading vehicle
    :param acc_param: dictionary with parameters of selected ACC controller
    :param acc_param_all_controllers: dictionary with parameter dictionaries for each ACC controller
    """
    # Load simulation parameter
    number_nodes_rrt = rrt_param.get("number_nodes")
    num_iterations = simulation_param.get("num_iterations")

    # Initialize vehicle planners
    lead_planner = RRTBackward(simulation_param, rrt_param, lead_vehicle_param)
    acc_planner = AccFactory.create(simulation_param, acc_vehicle_param, acc_param)

    # Initialize lists for nodes
    node_list = init_nodes_backward(acc_vehicle_param, lead_vehicle_param, simulation_param, rrt_param)

    # Initialization of variables for simulation
    valid_initial_node = False
    solution_node = None
    max_time_elapsed = False
    solution = False
    unsafe_node_reached = False

    # Start of simulation
    for idx in range(num_iterations):
        if simulation_param.get("verbose_mode"):
            print("Iteration: " + str(idx + 1))
        # Initialize node lists
        new_node_list = []
        node_calculation_time = timer()
        acc_state_list = acc_random_backward(node_list, acc_vehicle_param, simulation_param.get("dt"))

        while len(new_node_list) < number_nodes_rrt and not max_time_elapsed:
            # prevent dead end of backward search
            if abs(node_calculation_time - timer()) > simulation_param.get("max_comp_time_backward"):
                max_time_elapsed = True
                print("maximum time elapsed")
                break

            # Calculate lead vehicle backward state
            node = lead_planner.plan(node_list, acc_state_list, lead_vehicle_param, acc_vehicle_param,
                                     simulation_param)

            # Simulate ACC-vehicle forward based on sampled backward input
            a_acc = acc_planner.acc_control(node.lead_state.velocity, node.parent.acc_state.velocity,
                                            node.get_lead_x_position_rear(lead_vehicle_param),
                                            node.parent.get_acc_x_position_front(acc_vehicle_param),
                                            node.parent.acc_state.acceleration)
            inputs = [0, a_acc]
            state_acc = forward_propagation(node.parent.acc_state.x_position, node.parent.acc_state.y_position, 0,
                                            node.parent.acc_state.velocity, 0, inputs, simulation_param.get("dt"),
                                            simulation_param.get("dt"), acc_vehicle_param.get("dynamics_param"))

            new_acc_state = State(state_acc[0][0], state_acc[0][1], state_acc[0][2], state_acc[0][3], state_acc[0][4],
                                  0, a_acc, node.lead_state.time_step)
            node.add_acc_state(new_acc_state, lead_vehicle_param, acc_vehicle_param, simulation_param)

            # check for valid initial state and if unsafe state will be reached from this state
            unsafe_node_reached, valid_initial_node = check_unsafe_node_reached(node, simulation_param, rrt_param,
                                                                                lead_vehicle_param,
                                                                                acc_vehicle_param, acc_planner)
            if node.acc_state.velocity < 0 or not unsafe_node_reached:
                continue
            new_node_list.append(node)

            if unsafe_node_reached and valid_initial_node:
                print("valid initial node reached")
                solution_node = new_node_list[-1]
                solution = True
                break

        node_list = new_node_list

        # Evaluate if solution was found
        if unsafe_node_reached and valid_initial_node:
            break
        elif max_time_elapsed:
            break

    if simulation_param.get("store_results") or (unsafe_node_reached and valid_initial_node):
        if solution_node is None:
            solution_node = node_list[np.random.randint(0, len(node_list))]
        store_results(solution_node, simulation_param, acc_vehicle_param, lead_vehicle_param, rrt_param,
                      acc_param_all_controllers)

    return solution
