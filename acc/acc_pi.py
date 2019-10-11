"""
PI-controller based on Yanakiev and Kanellakopoulos: "Nonlinear Spacing Policies for Automated Heavy-Duty Vehicles".
"""
from acc.acc_interface import AccFactory
from common.utility_fcts import check_feasibility
from typing import Dict


class PIController(AccFactory):
    def __init__(self, simulation_param: Dict, acc_vehicle_param: Dict, acc_param: Dict):
        """
        :param simulation_param: parameters of the simulation environment
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param acc_param: dictionary with parameters of selected ACC controller
        """
        super().__init__(simulation_param, acc_vehicle_param)
        self._delta_s_min = acc_param.get("delta_s_min")
        self._k1 = acc_param.get("k1")
        self._k = acc_param.get("k")
        self._kp = acc_param.get("kp")
        self._ki = acc_param.get("ki")
        self._h0 = acc_param.get("h0")
        self._hc = acc_param.get("hc")

    def dynamic_headway(self, delta_v: float) -> float:
        """
        Calculates time-headway between two vehicles

        :param delta_v: relative velocity between leading and following vehicle [m/s]
        :return: time-headway for acceleration calculation [s]
        """
        time_headway = self._h0 - self._hc * delta_v
        if time_headway >= 1:
            time_headway = 1
        elif time_headway <= 0:
            time_headway = 0
        return time_headway

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

        time_headway = self.dynamic_headway(delta_v)
        s_desired = self._delta_s_min + time_headway * v_acc
        distance_error = delta_s - s_desired

        a_acc_new = (self._kp * (delta_v + self._k * distance_error)) + \
                    (self._ki * (1 / self._dt) * (delta_v + self._k * distance_error))

        a_acc_new = check_feasibility(a_acc_new, v_acc, self._a_min, self._a_max, self._dt, self._v_max, a_acc,
                                      self._j_min, self._j_max)

        return a_acc_new
