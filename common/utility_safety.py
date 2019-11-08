import numpy as np
from typing import List, Tuple
import sys


def simulate_vehicle_braking(velocity_list: List[float], position_list: List[float], acceleration: float,
                             a_min: float, dt: float, j_min: float) -> Tuple[List[float], List[float]]:
    """
    Forward simulation with kinematic single-track model

    :param velocity_list: list of previous velocity values of vehicle [m/s]
    :param position_list: list of previous x-position values of vehicle [m]
    :param acceleration: current acceleration of vehicle [rad]
    :param a_min: minimum acceleration of vehicle [m/s]
    :param j_min: minimum jerk of vehicle [rad]
    :param dt: time step size [s]
    :return: velocity and position list of braking vehicle
    """
    current_velocity = velocity_list[-1]
    current_position = position_list[-1]
    while current_velocity > 0:
        acceleration = max(a_min, acceleration + j_min * dt)
        if current_velocity + acceleration * dt >= 0:  # acceleration does
            # not lead to velocity < 0
            current_position = current_position + current_velocity * dt + 0.5 * acceleration * dt ** 2
            current_velocity = current_velocity + acceleration * dt
        else:  # acceleration leads to velocity < 0 -> prevent this
            delta_t_short = current_velocity / abs(acceleration)
            current_position = \
                current_position + current_velocity * delta_t_short + 0.5 * acceleration * delta_t_short ** 2

            current_velocity = 0
        position_list.append(current_position)
        velocity_list.append(current_velocity)

    return position_list, velocity_list


def safe_distance(v_acc: float, v_lead: float, a_min_acc: float, a_min_lead: float, t_react: float,
                  a_acc: float, a_lead: float, j_min_acc: float, j_min_lead: float, dt: float,
                  s_x_acc: float, s_x_lead: float, a_max_acc: float, j_max_acc: float) -> float:
    """
    Calculates safe distance between two vehicles, if both have the same negative acceleration

    :param v_acc: current velocity of ACC vehicle [m/s]
    :param v_lead: current velocity of leading vehicle [m/s]
    :param a_min_lead: minimum negative acceleration of leading vehicle [m/s^2]
    :param a_min_acc: minimum negative acceleration of ACC vehicle [m/s^2]
    :param a_max_acc: maximum positive acceleration of ACC vehicle [m/s^2]
    :param t_react: reaction time of ACC vehicle [s]
    :param a_acc: current acceleration of ACC vehicle [m/s^2]
    :param a_lead: current acceleration of leading vehicle [m/s^2]
    :param s_x_acc: current position of ACC vehicle [m/s^2]
    :param s_x_lead: current position of leading vehicle [m/s^2]
    :param j_min_acc: minimum negative jerk of ACC vehicle [m/s^3]
    :param j_min_lead: minimum negative acceleration of leading vehicle [m/s^3]
    :param j_max_acc: maximum positive jerk of ACC vehicle [m/s^3]
    :param dt: time step size [s]
    :return: safe distance between ACC vehicle and leading vehicle [m]
    """
    t = 0  # time step
    s_acc_list = [s_x_acc]
    v_acc_list = [v_acc]
    s_lead_list = [s_x_lead]
    v_lead_list = [v_lead]
    initial_distance = s_x_lead - s_x_acc

    if v_acc == 0:
        return 0.0

    # Leading vehicle braking:
    s_lead_list, v_lead_list = simulate_vehicle_braking(v_lead_list, s_lead_list, a_lead, a_min_lead, dt, j_min_lead)

    # ACC vehicle motion during reaction time:
    while t < t_react and v_acc > 0:
        if v_acc + min(a_max_acc, a_acc + j_max_acc * dt) * dt >= 0:
            a_acc = min(a_max_acc, a_acc + j_max_acc * dt)
            s_acc_curr = s_acc_list[-1] + v_acc * dt + 0.5 * a_acc * dt ** 2
            v_acc = v_acc + a_acc * dt
        else:
            a_acc = min(a_max_acc, a_acc + j_max_acc * dt)
            delta_t_short = v_acc / abs(a_acc)
            s_acc_curr = s_acc_list[-1] + v_acc * delta_t_short + 0.5 * a_acc * delta_t_short ** 2
            v_acc = 0
        s_acc_list.append(s_acc_curr)
        v_acc_list.append(v_acc)

        s_acc_curr = s_acc_list[-1] + v_acc * dt
        s_acc_list.append(s_acc_curr)
        t += dt

    # ACC vehicle braking:
    s_acc_list, v_acc_list = simulate_vehicle_braking(v_acc_list, s_acc_list, a_acc, a_min_acc, dt, j_min_acc)

    # Equalize list size:
    if len(s_acc_list) > len(s_lead_list):
        s_lead_list = s_lead_list + [s_lead_list[-1]] * (len(s_acc_list) - len(s_lead_list))
    elif len(s_lead_list) > len(s_acc_list):
        s_acc_list = s_acc_list + [s_acc_list[-1]] * (len(s_lead_list) - len(s_acc_list))

    # Safe distance calculation:
    delta_s = np.subtract(s_lead_list[1::], s_acc_list[1::])
    min_delta_s = np.min(delta_s)

    safe_dist = max(0, initial_distance - min_delta_s)

    return safe_dist


def unsafe_distance(v_acc: float, v_lead: float, a_min_acc: float, a_min_lead: float, a_acc: float, a_lead: float,
                    j_min_acc: float, j_min_lead: float, dt: float, s_x_acc: float, s_x_lead: float,
                    v_col: float) -> float:
    """
    Calculates safe distance between two vehicles, if both have the same negative acceleration

    :param v_acc: current velocity of ACC vehicle [m/s]
    :param v_lead: current velocity of leading vehicle [m/s]
    :param a_min_lead: minimum negative acceleration of leading vehicle [m/s^2]
    :param a_min_acc: minimum negative acceleration of ACC vehicle [m/s^2]
    :param a_acc: current acceleration of ACC vehicle [m/s^2]
    :param a_lead: current acceleration of leading vehicle [m/s^2]
    :param s_x_acc: current x-position of ACC vehicle front[m/s^2]
    :param s_x_lead: current x-position of leading vehicle rear[m/s^2]
    :param j_min_acc: minimum negative jerk of ACC vehicle [m/s^3]
    :param j_min_lead: minimum negative acceleration of leading vehicle [m/s^3]
    :param dt: time step size [s]
    :param v_col: minimum impact velocity of ACC vehicle [m/s]
    :return: unsafe distance between ACC vehicle and leading vehicle [m]
    """
    s_acc_list = [s_x_acc]
    v_acc_list = [v_acc]
    s_lead_list = [s_x_lead]
    v_lead_list = [v_lead]
    initial_distance = s_x_lead - s_x_acc

    if v_acc == 0:
        return 0.0

    # Leading vehicle braking:
    s_lead_list, v_lead_list = simulate_vehicle_braking(v_lead_list, s_lead_list, a_lead, a_min_lead, dt, j_min_lead)

    # ACC vehicle braking:
    s_acc_list, v_acc_list = simulate_vehicle_braking(v_acc_list, s_acc_list, a_acc, a_min_acc, dt, j_min_acc)

    # Equalize list size:
    if len(s_acc_list) > len(s_lead_list):
        s_lead_list = s_lead_list + [s_lead_list[-1]] * (len(s_acc_list) - len(s_lead_list))
        v_lead_list = v_lead_list + [0] * (len(v_acc_list) - len(v_lead_list))
    elif len(s_lead_list) > len(s_acc_list):
        s_acc_list = s_acc_list + [s_acc_list[-1]] * (len(s_lead_list) - len(s_acc_list))
        v_acc_list = v_acc_list + [0] * (len(v_lead_list) - len(v_acc_list))

    # Unsafe distance calculation:
    delta_s = np.subtract(s_lead_list[1::], s_acc_list[1::])
    min_delta_s = np.min(delta_s)
    if v_col > 0:
        # find first collision point
        s_offset = 0
        delta_s_index = None
        for idx, value in enumerate(delta_s):
            if value <= 0:
                delta_s_index = idx
                break
        # Unsafe distance is respected -> adjust position such that collision occurs:
        if delta_s_index is None:
            s_acc_list = [x + min_delta_s + 1e-12 for x in s_acc_list]
            delta_s = np.subtract(s_lead_list[1::], s_acc_list[1::])
            for idx, value in enumerate(delta_s):
                if value <= 0:
                    delta_s_index = idx
                    break
        # Matching of collision velocity to distance:
        while (v_lead_list[delta_s_index] - v_acc_list[delta_s_index]) > -v_col and s_acc_list[0] < s_lead_list[0]:
            s_acc_list = [x+0.10 for x in s_acc_list]
            s_offset += 0.10
            delta_s = np.subtract(s_lead_list[1::], s_acc_list[1::])
            for idx, value in enumerate(delta_s):
                if value <= 0:
                    delta_s_index = idx
                    break
        if (v_lead_list[delta_s_index] - v_acc_list[delta_s_index]) <= -v_col and s_acc_list[0] < s_lead_list[0]:
            unsafe_dist = max(0, initial_distance - s_offset)
        else:
            unsafe_dist = -sys.maxsize
    else:
        unsafe_dist = max(0, initial_distance - min_delta_s)

    return unsafe_dist
