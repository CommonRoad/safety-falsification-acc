"""
 Cascading PI controller based on ﻿H. Winner, S. Hakuli, and F. Lotz: "Handbuch Fahrerassistenzsysteme: Grundlagen,
 Komponenten und Systeme für aktive Sicherheit und Komfort".
"""
from acc.acc_interface import AccFactory
from common.utility_fcts import check_feasibility
from typing import Dict


class ExtendedPIControllerACC(AccFactory):
    def __init__(self, simulation_param: Dict, acc_vehicle_param: Dict, acc_param: Dict):
        """
        :param simulation_param: parameters of the simulation environment
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param acc_param: dictionary with parameters of selected ACC controller
        """
        super().__init__(simulation_param, acc_vehicle_param)
        self._kp = acc_param.get("kp")
        self._ki = acc_param.get("ki")
        self._tv = acc_param.get("tv")
        self._td = acc_param.get("td")
        self._t_set = acc_param.get("t_set")

    def acc_control(self, v_lead: float, v_acc: float, s_x_lead: float, s_x_acc: float, a_acc: float):
        """
        Calculates the ACC vehicle input

        :param v_lead: current velocity of leading vehicle [m/s]
        :param v_acc: current velocity of ACC vehicle [m/s]
        :param s_x_lead: x-position at the leading's vehicle rear [m]
        :param s_x_acc: x-position at the ACC vehicle's front [m]
        :param a_acc: current acceleration of the ACC vehicle [m/s²]
        :return: calculated acceleration [m/s²]
        """
        delta_s = s_x_lead - s_x_acc
        s_safe_approx = self._t_set * v_acc

        if delta_s > s_safe_approx:  # free flowing traffic
            delta_v = self._v_des - v_acc
            a_acc_new = delta_v/self._dt
        else:  # following traffic
            relative_velocity = v_lead - v_acc

            velocity_diff = (self._kp * (delta_s - s_safe_approx)) + \
                            (self._ki * 1/self._dt * (delta_s - s_safe_approx))
            a_acc_new = (self._kp * (relative_velocity + velocity_diff * 1/self._td)/self._tv) +\
                        (self._ki * 1/self._dt * (relative_velocity + velocity_diff * 1/self._td)/self._tv)

        a_acc_new = check_feasibility(a_acc_new, v_acc, self._a_min, self._a_max, self._dt, self._v_max, a_acc,
                                      self._j_min, self._j_max)

        return a_acc_new
