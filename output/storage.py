import pickle
import ruamel.yaml
from common.node import Node
from typing import Dict
from common.utility_fcts import get_date_and_time
import numpy as np
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.scenario.lanelet import Lanelet, LineMarking
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.planning.goal import GoalRegion, Interval, AngleInterval
from common.configuration import SearchType


def store_results(end_node: Node, simulation_param: Dict, acc_vehicle_param: Dict, lead_vehicle_param: Dict,
                  rrt_param: Dict, acc_param_complete: Dict):
    """
    Saving calculated lead vehicle trajectory and falsification configuration

    :param end_node: node of the last time step which is used for profile generation
    :param simulation_param: dictionary with simulation specific parameters
    :param acc_vehicle_param: dictionary with parameters of the ACC-equipped vehicle
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param rrt_param: dictionary with RRT-specific parameters
    :param acc_param_complete: dictionary with parameters of the ACC control algorithms
    """
    lead_inputs = []
    initial_node = None
    datetime = get_date_and_time()

    # Store initial node for backward search
    if simulation_param.get("search_type") == SearchType.BACKWARD:
        initial_node = end_node

    # Create lead vehicle acceleration profile
    while end_node.parent:
        lead_inputs.append([end_node.lead_state.steering_velocity, end_node.lead_state.acceleration])
        end_node = end_node.parent

    # Store initial node and reverse input profile for forward search and MCS
    if simulation_param.get("search_type") == SearchType.FORWARD \
            or simulation_param.get("search_type") == SearchType.MONTE_CARLO:
        lead_inputs.reverse()
        initial_node = end_node

    # Update initial configuration
    lead_vehicle_param["a_init"] = float(initial_node.lead_state.acceleration)
    lead_vehicle_param["v_init"] = float(initial_node.lead_state.velocity)
    lead_vehicle_param["s_x_init"] = float(initial_node.lead_state.x_position)
    lead_vehicle_param["initial_steering_angle"] = float(initial_node.lead_state.steering_angle)
    lead_vehicle_param["initial_yaw_angle"] = float(initial_node.lead_state.yaw_angle)

    acc_vehicle_param["a_init"] = float(initial_node.acc_state.acceleration)
    acc_vehicle_param["v_init"] = float(initial_node.acc_state.velocity)
    acc_vehicle_param["s_x_init"] = float(initial_node.acc_state.x_position)
    acc_vehicle_param["initial_steering_angle"] = float(initial_node.acc_state.steering_angle)
    acc_vehicle_param["initial_yaw_angle"] = float(initial_node.acc_state.yaw_angle)

    # Store input of leading vehicle
    with open("trajectory_" + datetime + ".pkl", "wb") as outfile:
        pickle.dump(lead_inputs, outfile)

    # Adjust output parameters
    acc_vehicle_param["controller"] = acc_vehicle_param.get("controller").value
    del acc_vehicle_param["dynamics_param"]
    del lead_vehicle_param["dynamics_param"]
    rrt_param["sampling_strategy"] = rrt_param.get("sampling_strategy").value
    simulation_param["search_type"] = simulation_param.get("search_type").value

    # Create one dictionary containing all other dictionaries to have a nicer formatting within the config file
    config = {"simulation_param": simulation_param,
              "rrt_param": rrt_param,
              "acc_vehicle_param": acc_vehicle_param,
              "lead_vehicle_param": lead_vehicle_param,
              "acc_param": acc_param_complete}

    # Store initial configuration
    with open("config_" + datetime + ".yaml", 'w') as outfile:
        ruamel.yaml.round_trip_dump(config, outfile, explicit_start=True)


def create_commonroad_scenario(current_node: Node, simulation_param: Dict, lead_vehicle_param: Dict,
                               acc_vehicle_param: Dict):
    """
    Creation of a CommonRoad scenario

    :param current_node: node of the last time step which is used for scenario generation
    :param simulation_param: dictionary with simulation specific parameters
    :param lead_vehicle_param: dictionary with physical parameters of the leading vehicle
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    """
    author = simulation_param.get("commonroad_scenario_author")
    affiliation = 'Technical University of Munich, Germany'
    source = 'Safety Falsification ACC'
    tags = simulation_param.get("commonroad_scenario_tags")
    lanelet_id = 2

    # Create initial/goal state and dynamic obstacle for leading vehicle
    state_list = []
    time_step = 1
    goal_position_shape = Rectangle(10, 3.5,
                                    np.array([current_node.lead_state.x_position -
                                              0.5 * lead_vehicle_param.get("dynamics_param").l - 5.01, 1.75]))
    goal_state = State(position=goal_position_shape,
                       velocity=Interval(0, 0.1),
                       orientation=AngleInterval(-0.01, 0.01), time_step=Interval(0, current_node.lead_state.time_step))
    final_x_position_lead = current_node.lead_state.x_position

    while current_node.parent:
        state = State(position=np.array([current_node.lead_state.x_position, current_node.lead_state.y_position]),
                      velocity=current_node.lead_state.velocity, steering_angle=0,  orientation=0,
                      time_step=current_node.lead_state.time_step, acceleration=current_node.lead_state.acceleration)
        state_list.append(state)
        current_node = current_node.parent
        time_step += 1

    initial_cr_state_lead = State(position=np.array([current_node.lead_state.x_position,
                                                     current_node.lead_state.y_position]),
                                  velocity=current_node.lead_state.velocity, steering_angle=0, orientation=0,
                                  time_step=0, acceleration=current_node.lead_state.acceleration, yaw_rate=0,
                                  slip_angle=0)

    state_list.reverse()
    initial_cr_state_acc = State(position=np.array([current_node.acc_state.x_position,
                                                    current_node.acc_state.y_position]),
                                 velocity=current_node.acc_state.velocity, steering_angle=0, orientation=0,
                                 time_step=0, acceleration=current_node.acc_state.acceleration, yaw_rate=0,
                                 slip_angle=0)
    trajectory = Trajectory(initial_time_step=1, state_list=state_list)
    shape = Rectangle(lead_vehicle_param.get("dynamics_param").l, lead_vehicle_param.get("dynamics_param").w)
    prediction = TrajectoryPrediction(trajectory, shape)
    lead_vehicle = DynamicObstacle(42, ObstacleType.CAR, shape, initial_cr_state_lead, prediction)

    # Create lanelet
    left_vertices_point_list = []
    center_vertices_point_list = []
    right_vertices_point_list = []
    for i in range(int(initial_cr_state_acc.position[0] - acc_vehicle_param.get("dynamics_param").l - 10),
                   int(final_x_position_lead + lead_vehicle_param.get("dynamics_param").l + 10)):
        left_vertices_point_list.append(np.array([i, 3.5]))
        center_vertices_point_list.append(np.array([i, 1.75]))
        right_vertices_point_list.append(np.array([i, 0.0]))

    left_vertices = np.array(left_vertices_point_list)
    center_vertices = np.array(center_vertices_point_list)
    right_vertices = np.array(right_vertices_point_list)
    lanelet = Lanelet(left_vertices, center_vertices, right_vertices, lanelet_id,
                      line_marking_left_vertices=LineMarking.SOLID, line_marking_right_vertices=LineMarking.SOLID)

    # Create scenario and planning problem
    scenario = Scenario(simulation_param.get("dt"), simulation_param.get("commonroad_benchmark_id"))
    scenario.lanelet_network.add_lanelet(lanelet)
    scenario.add_objects(lead_vehicle)
    goal_region = GoalRegion([goal_state])
    planning_problem = PlanningProblem(1, initial_cr_state_acc, goal_region)
    planning_problem_set = PlanningProblemSet([planning_problem])

    # write new scenario
    fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)
    filename = "./" + simulation_param.get("commonroad_benchmark_id") + ".xml"
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)
