"""
 Intelligent Driver Model ACC based on ﻿A. Kesting, M. Treiber, M. Schönhof, and D. Helbing: “Adaptive cruise control
 design for active congestion avoidance”.
"""
from acc.acc_interface import AccFactory
from common.utility_fcts import check_feasibility
import math
from typing import Dict


class IdmACC(AccFactory):
    def __init__(self, simulation_param: Dict, acc_vehicle_param: Dict, acc_param: Dict):
        """
        :param simulation_param: parameters of the simulation environment
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param acc_param: dictionary with parameters of selected ACC controller
        """
        super().__init__(simulation_param, acc_vehicle_param)
        self._delta_s_min = acc_param.get("delta_s_min")
        self._t_des = acc_param.get("t_des")
        self._b = acc_param.get("b")

    def acc_control(self, v_lead: float, v_acc: float, s_x_lead: float, s_x_acc: float, a_acc: float) -> float:
        """
        Calculates the ACC vehicle input based on the intelligent driver model ACC

        :param v_lead: current velocity of leading vehicle [m/s]
        :param v_acc: current velocity of ACC vehicle [m/s]
        :param s_x_lead: x-position at the leading's vehicle rear [m]
        :param s_x_acc: x-position at the ACC vehicle's front [m]
        :param a_acc: current acceleration of the ACC vehicle [m/s²]
        :return: calculated acceleration [m/s²]
        """
        delta_v = v_lead - v_acc
        delta_s = s_x_lead - s_x_acc

        desired_gap = \
            self._delta_s_min + v_acc * \
            self._t_des + ((v_acc * delta_v) / (2 * math.sqrt(self._a_max * self._b)))

        a_acc_new = self._a_max * (1 - (v_acc/self._v_des) ** 4) - ((desired_gap/delta_s) ** 2)

        a_acc_new = check_feasibility(a_acc_new, v_acc, self._a_min, self._a_max, self._dt, self._v_max, a_acc,
                                      self._j_min, self._j_max)

        return a_acc_new
