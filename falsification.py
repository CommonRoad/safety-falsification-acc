"""
Main file which starts simulation based on the configuration in config.yaml.
"""
from common.configuration import *
import lead_search.forward_search as forward
import lead_search.backward_search as backward
import lead_search.monte_carlo_simulation as monte_carlo


def main():
    # Initialization of variables for simulation
    config = load_yaml("config.yaml")
    simulation_param = create_sim_param(config.get("simulation_param"))
    lead_vehicle_param = create_lead_vehicle_param(config.get("lead_vehicle_param"))
    acc_vehicle_param = create_acc_vehicle_param(config.get("acc_vehicle_param"))
    rrt_param = create_rrt_param(config.get("rrt_param"))
    acc_param = create_acc_param(config.get("acc_param"), acc_vehicle_param.get("controller"))
    acc_param_all_controllers = config.get("acc_param")

    # Start search
    if simulation_param.get("search_type") == SearchType.FORWARD:
        forward.search(rrt_param, acc_vehicle_param, simulation_param, lead_vehicle_param, acc_param,
                       acc_param_all_controllers)
    elif simulation_param.get("search_type") == SearchType.BACKWARD:
        backward.search(rrt_param, acc_vehicle_param, simulation_param, lead_vehicle_param, acc_param,
                        acc_param_all_controllers)
    elif simulation_param.get("search_type") == SearchType.MONTE_CARLO:
        monte_carlo.search(rrt_param, acc_vehicle_param, simulation_param, lead_vehicle_param, acc_param,
                           acc_param_all_controllers)


if __name__ == "__main__":
    main()
