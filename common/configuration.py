from parameters_vehicle1 import parameters_vehicle1
from parameters_vehicle2 import parameters_vehicle2
from parameters_vehicle3 import parameters_vehicle3
import enum
from typing import Dict, Union
import ruamel.yaml


class SamplingStrategy(enum.Enum):
    """
    Enum describing different sampling strategies
    """
    LOCAL = "local"
    GLOBAL = "global"


class VehicleType(enum.Enum):
    """
    Enum describing vehicle types
    """
    LEAD = "lead_vehicle"
    ACC = "acc_vehicle"


class ACCController(enum.Enum):
    """
    Enum describing vehicle types
    """
    IDM = "idm"
    COLLISION_AVOIDANCE = "collision_avoidance"
    EXTENDED_PI = "extended_pi"
    PI = "pi"


class SearchType(enum.Enum):
    """
    Enum describing falsification search types
    """
    FORWARD = "forward"
    BACKWARD = "backward"
    MONTE_CARLO = "monte_carlo"


def create_sim_param(simulation_param: Dict) -> Dict:
    """
    Update simulation parameters

    :param simulation_param: dictionary with parameters of the simulation environment
    """
    if simulation_param.get("search_type") == "forward":
        simulation_param["search_type"] = SearchType.FORWARD
    elif simulation_param.get("search_type") == "backward":
        simulation_param["search_type"] = SearchType.BACKWARD
    elif simulation_param.get("search_type") == "monte_carlo":
        simulation_param["search_type"] = SearchType.MONTE_CARLO
    else:
        raise ValueError('Wrong search type in config file defined.')

    return simulation_param


def create_rrt_param(rrt_param: Dict) -> Dict:
    """
    Update RRT parameters

    :param rrt_param: dictionary with RRT parameters
    """
    if rrt_param.get("sampling_strategy") == "local":
        rrt_param["sampling_strategy"] = SamplingStrategy.LOCAL
    elif rrt_param.get("sampling_strategy") == "global":
        rrt_param["sampling_strategy"] = SamplingStrategy.GLOBAL
    else:
        raise ValueError('Wrong sampling strategy in config file defined.')

    return rrt_param


def create_acc_vehicle_param(acc_vehicle_param: Dict) -> Dict:
    """
    Update ACC vehicle parameters

    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    """
    if acc_vehicle_param.get("controller") == "idm":
        acc_vehicle_param["controller"] = ACCController.IDM
    elif acc_vehicle_param.get("controller") == "collision_avoidance":
        acc_vehicle_param["controller"] = ACCController.COLLISION_AVOIDANCE
    elif acc_vehicle_param.get("controller") == "extended_pi":
        acc_vehicle_param["controller"] = ACCController.EXTENDED_PI
    elif acc_vehicle_param.get("controller") == "pi":
        acc_vehicle_param["controller"] = ACCController.PI
    else:
        raise ValueError('Wrong ACC controller in config file defined.')

    if acc_vehicle_param.get("vehicle_number") == 1:
        acc_vehicle_param["dynamics_param"] = parameters_vehicle1()
    elif acc_vehicle_param.get("vehicle_number") == 2:
        acc_vehicle_param["dynamics_param"] = parameters_vehicle2()
    elif acc_vehicle_param.get("vehicle_number") == 3:
        acc_vehicle_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError('Wrong vehicle number for ACC vehicle in config file defined.')

    return acc_vehicle_param


def create_lead_vehicle_param(lead_vehicle_param: Dict) -> Dict:
    """
    Update lead vehicle parameters

    :param lead_vehicle_param: dictionary with physical parameters of the leading vehicle
    """
    if lead_vehicle_param.get("vehicle_number") == 1:
        lead_vehicle_param["dynamics_param"] = parameters_vehicle1()
    elif lead_vehicle_param.get("vehicle_number") == 2:
        lead_vehicle_param["dynamics_param"] = parameters_vehicle2()
    elif lead_vehicle_param.get("vehicle_number") == 3:
        lead_vehicle_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError('Wrong vehicle number for leading vehicle in config file defined.')

    return lead_vehicle_param


def create_acc_param(controller_param: Dict, controller: ACCController) -> Dict:
    """
    Update ACC controller parameters

    :param controller_param: list of dictionaries with ACC parameter
    :param controller: enum value for ACC controller
    """
    name_selected_controller = controller.value
    for controller_name, controller in controller_param.items():
        if controller_name == name_selected_controller:
            return controller

    raise ValueError('Matching of controller parameters to defined controller not possible.')


def load_yaml(file_name: str) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    with open(file_name, 'r') as stream:
        try:
            config = ruamel.yaml.round_trip_load(stream, preserve_quotes=True)
            return config
        except ruamel.yaml.YAMLError as exc:
            print(exc)
            raise ruamel.yaml.YAMLError

