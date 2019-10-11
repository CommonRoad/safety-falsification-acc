from common.node import Node
from common.state import State
from common.utility_fcts import forward_simulation, func_ks, reaction_delay
from init_KS import init_KS
import numpy as np
from scipy.integrate import odeint
from typing import Dict, List, Tuple


def init_nodes_forward(rrt_param: Dict, acc_vehicle_param: Dict, simulation_param: Dict,
                       lead_vehicle_param: Dict) -> List[Node]:
    """
    Initialize list of nodes for forward search

    :param rrt_param: dictionary with RRT parameters
    :param acc_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :return: list of initial nodes
    """
    node_list = []
    while len(node_list) < rrt_param.get("number_nodes"):
        # Initialize ACC vehicle state with random velocity in range [0, v_max]
        acc_state = State(acc_vehicle_param.get("s_init"), 1.75, 0,
                          np.random.uniform(simulation_param.get("v_col"),
                                            acc_vehicle_param.get("dynamics_param").longitudinal.v_max), 0, 0, 0, 0)

        # Initialize leading vehicle state with random distance > safe distance and random velocity in range [0, v_max]
        lead_state = State(np.random.uniform(acc_vehicle_param.get("s_init") +
                                             acc_vehicle_param.get("dynamics_param").l / 2 +
                                             simulation_param.get("delta_s_init_min"),
                                             acc_vehicle_param.get("s_init") +
                                             acc_vehicle_param.get("dynamics_param").l / 2 +
                                             lead_vehicle_param.get("dynamics_param").l / 2 +
                                             simulation_param.get("delta_s_init_max")), 1.75, 0,
                           np.random.uniform(0, lead_vehicle_param.get("dynamics_param").longitudinal.v_max),
                           0, 0, 0, 0)
        node = Node(None, lead_state, acc_state, lead_vehicle_param, acc_vehicle_param, simulation_param)

        # Evaluate if node corresponds to constraints:
        if node.delta_s > node.safe_distance:
            node_list.append(node)
    return node_list


def is_valid_final_node(node: Node) -> bool:
    """
    Check if unsafe node is in valid range

    :param node: node to validate
    :return: True or False
    """
    if node.delta_s <= node.unsafe_distance:
        return True
    else:
        return False


def check_feasibility_backward(a_new: float, v: float, a_min: float, a_max: float,
                               dt: float, v_max: float, a_cur: float, j_min: float,
                               j_max: float) -> float:
    """
    Limits vehicle acceleration to maximum/minimum value - considers velocity and jerk limit

    :param a_new: calculated acceleration to reach goal [m/s^2]
    :param v: current velocity of vehicle [m/s]
    :param a_min: minimum acceleration of vehicle [m/s^2]
    :param a_max: maximum acceleration of vehicle [m/s^2]
    :param dt: time step size [s]
    :param v_max: maximum possible/allowed velocity of the vehicle [m/s]
    :param a_cur: current acceleration of the vehicle [m/s^2]
    :param j_min: minimum jerk of the vehicle [m/s^3]
    :param j_max: maximum jerk of the vehicle [m/s^3]
    :return: Bounded acceleration as scalar value [m/s^2]
    """
    # Limit jerk
    if (a_new - a_cur) > j_max * dt:
        a_new = a_cur + j_max * dt
    elif (a_new - a_cur) < j_min * dt:
        a_new = a_cur + j_min * dt

    # Limit acceleration
    if a_new < a_min:
        a_new = a_min
    elif a_new > a_max:
        a_new = a_max

    # Check if acceleration would lead to negative velocity
    velocity_change = dt * abs(a_new)
    if a_new > 0 and v < velocity_change:
        a_new = (v / dt)

    # Check if acceleration would lead to larger velocity than possible
    if a_new < 0 and v + velocity_change > v_max:
        a_new = - (v_max - v) / dt

    return a_new


def backward_simulation(x_position_center: float, y_position_center: float, steering_angle: float, velocity: float,
                        yaw_angle: float, inputs: List[float], end_time: float, dt: float,
                        vehicle_parameter: Dict) -> Tuple[np.ndarray, dict]:
    """
    Backward simulation with kinematic single-track model

    :param x_position_center: x-position of the vehicle center [m]
    :param y_position_center: y-position of the vehicle center [m]
    :param steering_angle: steering angle of the vehicle [rad]
    :param velocity: velocity of the vehicle [m/s]
    :param yaw_angle: yaw angle of the vehicle [rad]
    :param inputs: inputs [steering velocity, acceleration] of the vehicle [rad/s, m/s^2]
    :param end_time: end time for the backward simulation [s]
    :param dt: time step [s]
    :param vehicle_parameter: vehicle parameter vector
    :return: right-hand side of differential equations
    """
    initial_state = [x_position_center, y_position_center, steering_angle, velocity, yaw_angle, 0, 0]

    x0_ks = init_KS(initial_state)
    t = np.arange(0, end_time + dt, dt)
    x = odeint(func_ks, x0_ks, t, args=(inputs, vehicle_parameter))
    x[1][0] = x[0][0] - abs(x[1][0] - x[0][0])
    x[1][3] = x[0][3] - abs(x[1][3] - x[0][3])

    return x
