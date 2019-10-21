"""
Execution of stored results.
You must provide an input parameter to the script (e.g., "python review.py xyz", where the solution files are called
config_xzy.yaml and trajectory_xyz.pkl).
"""
from common.node import Node
from acc.acc_interface import AccFactory
from lead_search.rrt_forward import RRTForward
from common.utility_fcts import check_feasibility, acc_input_forward
from common.state import State
from output.storage import create_commonroad_scenario
from output.visualization import plot_figures
from common.configuration import *
import sys
import pickle


def main():
    try:
        solution_file_suffix = sys.argv[1]
    except IndexError:
        print("Solution file suffix missing!")
        raise ValueError

    # load configuration and trajectory of leading vehicle
    config = load_yaml("config_" + solution_file_suffix + ".yaml")
    simulation_param = create_sim_param(config.get("simulation_param"))
    lead_vehicle_param = create_lead_vehicle_param(config.get("lead_vehicle_param"))
    acc_vehicle_param = create_acc_vehicle_param(config.get("acc_vehicle_param"))
    rrt_param = create_rrt_param(config.get("rrt_param"))
    acc_param = create_acc_param(config.get("acc_param"), acc_vehicle_param.get("controller"))

    with open("trajectory_" + solution_file_suffix + ".pkl", 'rb') as file:
        lead_inputs = pickle.load(file)

    # Initialize vehicle planners
    lead_planner = RRTForward(simulation_param, rrt_param, lead_vehicle_param)
    acc_planner = AccFactory.create(simulation_param, acc_vehicle_param, acc_param)

    # Initialize states and node list
    node_list = []
    acc_state = State(acc_vehicle_param.get("s_init"), 1.75, 0, acc_vehicle_param.get("v_init"), 0, 0,
                      acc_vehicle_param.get("a_init"), 0)
    lead_state = State(lead_vehicle_param.get("s_init"), 1.75, 0, lead_vehicle_param.get("v_init"), 0, 0,
                       lead_vehicle_param.get("a_init"), 0)
    current_node = Node(None, lead_state, acc_state, lead_vehicle_param, acc_vehicle_param, simulation_param)

    # Initialization of variables for simulation
    collision = False
    unsafe = False
    node_list.append(current_node)
    t_react_acc = int(acc_vehicle_param.get("t_react") / simulation_param.get("dt"))

    # Start of simulation
    for idx in range(len(lead_inputs)):
        # Plan the ACC trajectory based on the selected controller
        acc_state = acc_input_forward(acc_planner, current_node, lead_vehicle_param,
                                      acc_vehicle_param, t_react_acc, simulation_param.get("dt"))

        # Forward simulation of the leading vehicle with predefined acceleration
        acceleration = check_feasibility(lead_inputs[idx][1], current_node.lead_state.velocity,
                                         lead_vehicle_param.get("a_min"),
                                         lead_vehicle_param.get("a_max"), simulation_param.get("dt"),
                                         acc_vehicle_param.get("dynamics_param").longitudinal.v_max,
                                         current_node.lead_state.acceleration,
                                         lead_vehicle_param.get("j_min"), lead_vehicle_param.get("j_max"))

        new_node = lead_planner.forward_simulation(lead_inputs[idx][0], acceleration, current_node, acc_state,
                                                   lead_vehicle_param, acc_vehicle_param,  simulation_param)

        # Check if collision occurred, if yes stop simulation
        current_node = new_node
        node_list.append(current_node)

        if new_node.delta_s <= new_node.unsafe_distance:
            unsafe = True
            print("unsafe distance")
        if new_node.delta_s <= 0:
            collision = True
            print("collision")
            break

    # If lead profile terminates in unsafe state, continue with full braking
    while unsafe and not collision:
        # Plan the ACC trajectory based on the selected controller
        acc_state = acc_input_forward(acc_planner, current_node, lead_vehicle_param,
                                      acc_vehicle_param, t_react_acc, simulation_param.get("dt"))

        # Forward simulation of the lead vehicle with minimum acceleration
        acceleration = check_feasibility(lead_vehicle_param.get("a_min"), current_node.lead_state.velocity,
                                         lead_vehicle_param.get("a_min"),
                                         lead_vehicle_param.get("a_max"), simulation_param.get("dt"),
                                         lead_vehicle_param.get("dynamics_param").longitudinal.v_max,
                                         current_node.lead_state.acceleration, lead_vehicle_param.get("j_min"),
                                         lead_vehicle_param.get("j_max"))

        new_node = lead_planner.forward_simulation(0, acceleration, current_node, acc_state,
                                                   lead_vehicle_param, acc_vehicle_param, simulation_param)

        # Check if collision occurred, if yes stop simulation
        current_node = new_node
        node_list.append(current_node)
        if new_node.delta_s <= 0:
            print("collision")
            break

    # Print results / create CommonRoad scenario / store videos
    create_commonroad_scenario(current_node, simulation_param, lead_vehicle_param, acc_vehicle_param)
    plot_figures([current_node], simulation_param, lead_vehicle_param, acc_vehicle_param)


if __name__ == "__main__":
    main()
