"""
ACC-controller with integrated collision avoidance based on Mullakkal-Babu et al.: "Design and analysis of
Full Range Adaptive Cruise Control with integrated collision avoidance strategy".
"""
from acc.acc_interface import AccFactory
from common.utility_fcts import check_feasibility
import numpy as np
from typing import Dict


class CAACC(AccFactory):
    def __init__(self, simulation_param: Dict, acc_vehicle_param: Dict, acc_param: Dict):
        """
        :param simulation_param: parameters of the simulation environment
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param acc_param: dictionary with parameters of selected ACC controller
        """
        super().__init__(simulation_param, acc_vehicle_param)
        self._delta_s_min = acc_param.get("delta_s_min")
        self._k1 = acc_param.get("k1")
        self._k2 = acc_param.get("k2")
        self._P = acc_param.get("P")
        self._Q = acc_param.get("Q")
        self._t_des = acc_param.get("t_des")

    def _get_spacing_error(self, delta_s: float, v_acc: float):
        """
        Calculates error for distance to leading vehicle

        :param delta_s: distance between leading vehicle and ACC vehicle [m]
        :param v_acc: current velocity of ACC vehicle [m/s]
        :return: error for spacing between vehicles [m]
        """
        error1 = delta_s - self._delta_s_min - v_acc * self._t_des
        error2 = (self._v_des - v_acc) * self._t_des
        return min(error1, error2)

    def _error_response(self, delta_s: float):
        """
        Calculates error for response of ACC vehicle (sigmoid function)

        :param delta_s: distance between leading vehicle and ACC vehicle [m]
        :return: error in dependence of distance
        """
        denominator = 1 + self._Q * np.exp(-delta_s / self._P)
        return (-1 / denominator) + 1

    def acc_control(self, v_lead: float, v_acc: float, s_x_lead: float, s_x_acc: float, a_acc: float) -> float:
        """
        Calculates the ACC vehicle input

        :param v_lead: current velocity of leading vehicle [m/s]
        :param v_acc: current velocity of ACC vehicle [m/s]
        :param s_x_lead: x-position at the leading's vehicle rear [m]
        :param s_x_acc: x-position at the ACC vehicle's front [m]
        :param a_acc: current acceleration of the ACC vehicle [m/s²]
        :return: calculated acceleration [m/s²]
        """
        delta_v = v_lead - v_acc
        delta_s = s_x_lead - s_x_acc

        a_acc_new = \
            self._k1 * self._get_spacing_error(delta_s, v_acc) + \
            self._k2 * delta_v * self._error_response(delta_s)

        a_acc_new = check_feasibility(a_acc_new, v_acc, self._a_min, self._a_max, self._dt, self._v_max, a_acc,
                                      self._j_min, self._j_max)

        return a_acc_new
