from abc import ABC, abstractmethod
from typing import Dict
from common.configuration import ACCController


class AccFactory(ABC):
    def __init__(self, simulation_param: Dict, acc_vehicle_param: Dict):
        """
        :param simulation_param: parameters of the simulation environment
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        """
        self._dt = simulation_param.get("dt")
        self._a_max = acc_vehicle_param.get("a_max")
        self._a_min = acc_vehicle_param.get("a_min")
        self._j_max = acc_vehicle_param.get("j_max")
        self._j_min = acc_vehicle_param.get("j_min")
        self._v_max = acc_vehicle_param.get("dynamics_param").longitudinal.v_max
        self._fov = acc_vehicle_param.get("sensor_detection_range")
        self._vehicle = acc_vehicle_param.get("dynamics_param")
        self._v_des = acc_vehicle_param.get("v_des")

    @classmethod
    def create(cls, simulation_param: Dict, acc_vehicle_param: Dict, acc_param: Dict):
        """
        Initializes desired ACC algorithm

        :param simulation_param: parameters of the simulation environment
        :param acc_vehicle_param: physical parameters of the ACC vehicle
        :param acc_param: dictionary with parameters of selected ACC controller
        :return: object of selected ACC controller
        """
        acc_type = acc_vehicle_param.get("controller")
        if acc_type == ACCController.PI:
            from acc.acc_pi import PIController
            return PIController(simulation_param, acc_vehicle_param, acc_param)
        elif acc_type == ACCController.COLLISION_AVOIDANCE:
            from acc.acc_collision_avoidance import CAACC
            return CAACC(simulation_param, acc_vehicle_param, acc_param)
        elif acc_type == ACCController.IDM:
            from acc.acc_intelligent_driver_model import IdmACC
            return IdmACC(simulation_param, acc_vehicle_param, acc_param)
        elif acc_type == ACCController.EXTENDED_PI:
            from acc.acc_extended_pi import ExtendedPIControllerACC
            return ExtendedPIControllerACC(simulation_param, acc_vehicle_param, acc_param)

    @abstractmethod
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
        pass
