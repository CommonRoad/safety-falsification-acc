from init_KS import init_KS
from vehicleDynamics_KS import vehicleDynamics_KS
from common.node import Node
import numpy as np
from scipy.integrate import odeint
from typing import List, Tuple, Dict
from datetime import datetime
from common.state import State


def func_ks(x: List[float], t: float, u: List[float], p: List[float]) -> List[float]:
    """
    Vehicle dynamics for kinematic single-track model

    :param x: vehicle state vector
    :param t: time parameter (used in scipy.odeint)
    :param u: vehicle input vector
    :param p: vehicle parameter vector
    :return: right-hand side of differential equations
    """
    f = vehicleDynamics_KS(x, u, p)
    return f


def forward_propagation(x_position_center: float, y_position_center: float, steering_angle: float, velocity: float,
                        yaw_angle: float, inputs: List[float], end_time: float, dt: float,
                        vehicle_parameter: float) -> Tuple[np.ndarray, dict]:
    """
    Forward propagation with kinematic single-track model

    :param x_position_center: x-position of the vehicle center [m]
    :param y_position_center: y-position of the vehicle center [m]
    :param steering_angle: steering angle of the vehicle [rad]
    :param velocity: velocity of the vehicle [m/s]
    :param yaw_angle: yaw angle of the vehicle [rad]
    :param inputs: inputs [steering velocity, acceleration] of the vehicle [rad/s, m/s^2]
    :param end_time: end time for the forward simulation [s]
    :param dt: time step size [s]
    :param vehicle_parameter: vehicle parameter vector
    :return: right-hand side of differential equations
    """
    initial_state = [x_position_center, y_position_center, steering_angle, velocity, yaw_angle, 0, 0]

    x0_ks = init_KS(initial_state)
    t = np.arange(0, end_time + dt, dt)
    x = odeint(func_ks, x0_ks, t, args=(inputs, vehicle_parameter))

    # reset small negative velocity to zero
    if -1e-9 < x[1][3] < 0:
        x[1][3] = 0.0
    return x


def check_feasibility(a_new: float, v: float, a_min: float, a_max: float, dt: float, v_max: float, a_cur: float,
                      j_min: float, j_max: float) -> float:
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
    :return: bounded acceleration value [m/s^2]
    """
    if a_new < 0 and v == 0.0:
        return 0.0

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
    if a_new < 0 and v < velocity_change:
        a_new = -(v / dt)

    # Check if acceleration would lead to larger velocity than possible
    if a_new > 0 and v + velocity_change > v_max:
        a_new = (v_max - v) / dt

    # If acceleration is very close to zero
    return a_new


def get_date_and_time() -> str:
    """
    Returns current data and time

    :return: Current date and time as string
    """
    current_time = datetime.now().time()
    current_time = str(current_time)
    current_time = current_time.replace(':', ' _')
    current_time = current_time.replace('.', ' _')
    current_date = str(datetime.now().day) + "_" + str(datetime.now().month) + "_" + str(datetime.now().year)

    return current_date + "_" + current_time


def acc_input_forward(acc_planner, node, lead_vehicle_param, acc_vehicle_param, t_react_acc, dt) -> State:
    """
    Initialize list of nodes for forward search

    :param acc_planner: ACC planning algorithm
    :param node: parent node
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    :param t_react_acc: number time steps of reaction time delay
    :param dt: time step size [s]
    :return: new ACC state
    """
    # Plan the ACC trajectory based on the selected controller
    a_acc = acc_planner.acc_control(node.lead_state.velocity, node.acc_state.velocity,
                                    node.get_lead_x_position_rear(lead_vehicle_param),
                                    node.get_acc_x_position_front(acc_vehicle_param), node.acc_state.acceleration)
    # Integrate reaction time
    inputs = reaction_delay(node, t_react_acc, a_acc, dt, acc_vehicle_param)

    # forward simulation
    sim_acc_state = forward_propagation(node.acc_state.x_position, node.acc_state.y_position, 0,
                                        node.acc_state.velocity, 0, inputs, dt, dt,
                                        acc_vehicle_param.get("dynamics_param"))

    new_acc_state = State(sim_acc_state[-1][0], sim_acc_state[-1][1], sim_acc_state[-1][2], sim_acc_state[-1][3],
                          sim_acc_state[-1][4], 0, a_acc, node.acc_state.time_step + 1)

    return new_acc_state


def reaction_delay(node: Node, t_react_steps: int, acceleration: float, dt: float,
                   acc_vehicle_param: Dict) -> List[float]:
    """
    Function delays input of vehicle for provided number of time steps

    :param node: parent node
    :param t_react_steps: number of time steps the input is delayed
    :param acceleration: calculated acceleration at current time step
    :param dt: time step size
    :param acc_vehicle_param: dictionary with physical parameters of the lead vehicle
    :return: delayed input for vehicle
    """
    if t_react_steps > 0:
        delayed_node = node
        delayed_acceleration = acceleration
        for i in range(t_react_steps):
            if delayed_node.parent:
                delayed_acceleration = delayed_node.acc_state.acceleration
                delayed_node = delayed_node.parent
            else:
                delayed_acceleration = delayed_node.acc_state.acceleration

        # additional feasibility check necessary for reaction delay
        a_acc_new = check_feasibility(delayed_acceleration, node.acc_state.velocity, acc_vehicle_param.get("a_min"),
                                      acc_vehicle_param.get("a_max"), dt,
                                      acc_vehicle_param.get("dynamics_param").longitudinal.v_max,
                                      node.acc_state.acceleration, acc_vehicle_param.get("j_min"),
                                      acc_vehicle_param.get("j_max"))
    else:
        a_acc_new = acceleration

    return [0, a_acc_new]
