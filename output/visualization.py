import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import imageio
from operator import sub
from common.utility_fcts import get_date_and_time
from common.node import Node
from typing import List, Dict, Tuple, Union
import matplotlib as mp
mp.use('Qt5Agg')


def create_profiles(node_list: List[Node], simulation_param: Dict, lead_vehicle_param: Dict, acc_vehicle_param: Dict) \
        -> Tuple[Union[np.ndarray, List[float]], List[float], List[float], List[float],
                 List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Plotting of positions, acceleration, velocity, and distance of ACC and lead vehicle, respectively

    :param node_list: list of nodes from the last time step
    :param simulation_param: dictionary with  simulation-specific parameters
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param acc_vehicle_param: dictionary with physical parameters of the ACC-equipped vehicle
    """
    # Initialization
    dt = simulation_param.get("dt")
    if len(node_list) is 1:
        end_node = node_list[0]
    else:
        end_node = node_list[np.random.randint(0, len(node_list) - 1)]

    velocity_profile_lead = []
    velocity_profile_acc = []
    acceleration_profile_lead = []
    acceleration_profile_acc = []
    x_position_profile_lead = []
    x_position_profile_acc = []
    y_position_profile_lead = []
    y_position_profile_acc = []
    x_distance_profile = []
    unsafe_distance_profile = []
    safe_distance_profile = []

    # Iterate backwards and add information to lists
    while end_node.parent:
        velocity_profile_acc.append(end_node.acc_state.velocity)
        x_position_profile_acc.append(end_node.get_acc_x_position_front(acc_vehicle_param))
        y_position_profile_acc.append(end_node.acc_state.y_position)
        acceleration_profile_acc.append(end_node.acc_state.acceleration)
        velocity_profile_lead.append(end_node.lead_state.velocity)
        x_position_profile_lead.append(end_node.get_lead_x_position_rear(lead_vehicle_param))
        y_position_profile_lead.append(end_node.lead_state.y_position)
        acceleration_profile_lead.append(end_node.lead_state.acceleration)
        x_distance_profile.append(end_node.delta_s)
        unsafe_distance_profile.append(max(0, end_node.unsafe_distance))
        safe_distance_profile.append(end_node.safe_distance)
        end_node = end_node.parent

    # Add initial state
    velocity_profile_acc.append(end_node.acc_state.velocity)
    x_position_profile_acc.append(end_node.get_acc_x_position_front(acc_vehicle_param))
    y_position_profile_acc.append(end_node.acc_state.y_position)
    acceleration_profile_acc.append(end_node.acc_state.acceleration)
    velocity_profile_lead.append(end_node.lead_state.velocity)
    x_position_profile_lead.append(end_node.get_lead_x_position_rear(lead_vehicle_param))
    y_position_profile_lead.append(end_node.lead_state.y_position)
    acceleration_profile_lead.append(end_node.lead_state.acceleration)
    x_distance_profile.append(end_node.delta_s)
    unsafe_distance_profile.append(max(0, end_node.unsafe_distance))
    safe_distance_profile.append(end_node.safe_distance)

    # Create time array
    time = np.linspace(0, len(x_position_profile_lead) * dt - dt, num=len(x_position_profile_lead), endpoint=True)

    # reverse array
    velocity_profile_lead = list(reversed(velocity_profile_lead))
    velocity_profile_ego = list(reversed(velocity_profile_acc))
    x_position_profile_lead = list(reversed(x_position_profile_lead))
    x_position_profile_ego = list(reversed(x_position_profile_acc))
    acceleration_profile_lead = list(reversed(acceleration_profile_lead))
    acceleration_profile_ego = list(reversed(acceleration_profile_acc))
    x_distance_profile = list(reversed(x_distance_profile))
    safe_distance = list(reversed(safe_distance_profile))
    unsafe_distance = list(reversed(unsafe_distance_profile))

    return time, velocity_profile_lead, velocity_profile_ego, x_position_profile_lead, x_position_profile_ego, \
           acceleration_profile_lead, acceleration_profile_ego, x_distance_profile, safe_distance, unsafe_distance


def plot_figures(node_list: List[Node], simulation_param: Dict, lead_vehicle_param: Dict, acc_vehicle_param: Dict):
    """
    Plotting of positions, acceleration, velocity, and distance of ACC and leading vehicle, respectively

    :param node_list: list of nodes from the last time step
    :param simulation_param: dictionary with  simulation-specific parameters
    :param lead_vehicle_param: dictionary with physical parameters of the leading vehicle
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    """
    simulation_type = simulation_param.get("search_type").value
    acc_controller = acc_vehicle_param.get("controller").value

    # Storage related configuration
    width = 3.39
    height = width * (math.sqrt(5) - 1.0) / 2.0
    figsize = [width, height]
    linewidth_plot = 0.75
    mp.rcParams.update({'font.size': 9})
    mp.rcParams.update({'axes.linewidth': 0.25})
    mp.rcParams.update({'figure.autolayout': True})
    mp.rcParams.update({'legend.frameon': False})

    # Directory creation if plots should be stored
    datetime = get_date_and_time()
    if not os.path.exists("figures/"):
        os.mkdir("figures/")
    path = "figures/" + acc_controller + "_" + simulation_type + "_" + datetime
    os.mkdir(path)

    # Create profiles
    time, velocity_profile_lead, velocity_profile_acc, x_position_profile_lead, x_position_profile_acc, \
    acceleration_profile_lead, acceleration_profile_acc, x_distance_profile, safe_distance, unsafe_distance = \
        create_profiles(node_list, simulation_param, lead_vehicle_param, acc_vehicle_param)

    # Velocity plot
    plt.figure(1, figsize=figsize)
    plt.ylabel(r'$v~[m/s]$')
    plt.xlabel(r'$t~[s]$')
    plt.plot(time, velocity_profile_lead, "-", color=(0.0, 0.0, 0.5, 1), label=r'$v_{lead}$', linewidth=linewidth_plot)
    plt.plot(time, velocity_profile_acc, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$v_{acc}$', linewidth=linewidth_plot)
    plt.legend(loc='best')
    plt.savefig(path + "/velocity_" + acc_controller + "_" + simulation_type + ".svg", format="svg")

    # Acceleration plot
    plt.figure(2, figsize=figsize)
    plt.ylabel(r'$a~[m/s^2]$')
    plt.xlabel(r'$t~[s]$')
    plt.ylim([-8.5, 2])
    plt.step(time, acceleration_profile_lead, color=(0.0, 0.0, 0.5, 1), label=r'$a_{lead}$', linewidth=linewidth_plot)
    plt.step(time, acceleration_profile_acc, color=(0.3, 0.3, 0.3, 0.35), label=r'$a_{acc}$', linewidth=linewidth_plot)
    plt.legend(loc='best')
    plt.savefig(path + "/acceleration_" + acc_controller + "_" + simulation_type + ".svg", format="svg")

    # Distance plot
    plt.figure(3, figsize=figsize)
    plt.ylabel(r'$\Delta s~[m]$')
    plt.xlabel(r'$t~[s]$')
    plt.plot(time, x_distance_profile, "-", color=(0.75, 0.75, 0.25, 1), label=r'$\Delta s$', linewidth=linewidth_plot)
    plt.plot(time, safe_distance, "-", color='g', label=r'$s_{safe}$', linewidth=linewidth_plot)
    plt.plot(time, unsafe_distance, "-", color=(1, 0, 0, 1), label=r'$s_{unsafe}$', linewidth=linewidth_plot)
    plt.legend(loc='best')
    plt.savefig(path + "/distance_" + acc_controller + "_" + simulation_type + ".svg", format="svg")
    plt.show()


def animate_profiles(number_frame: int, lead_vehicle_profile: List[float], acc_profile: List[float],
                     time_profile: List[float], x_label: str, y_label: str, x_min: float,
                     x_max: float, y_min: float, y_max: float):
    """
    Creation of GIF based on profiles of leading and ACC vehicle

    :param number_frame: current time step
    :param lead_vehicle_profile: profile of leading vehicle
    :param acc_profile: profile of ACC vehicle
    :param time_profile: list with all time steps
    :param x_label: label of x-axis
    :param y_label: label of y-axis
    :param x_min: minimum value of x-axis
    :param x_max: maximum value of x-axis
    :param y_min: minimum value of y-axis
    :param y_max: maximum value of y-axis
    """
    linewidth_plot = 1.75
    lead_vehicle_short = lead_vehicle_profile[:number_frame + 1]
    acc_short = acc_profile[:number_frame + 1]
    time_short = time_profile[:number_frame + 1]
    plt.rcParams.update({'font.size': 14})
    plt.cla()
    fig = plt.figure(1, figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.xaxis.set_label_coords(0.5, -0.075)

    plt.plot(time_short, lead_vehicle_short, color=(0.0, 0.0, 0.4, 0.75), label='$lead vehicle',
             linewidth=linewidth_plot)
    plt.plot(time_short, acc_short, color=(0.3, 0.3, 0.3, 0.75), label='$ACC$', linewidth=linewidth_plot)
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.xlabel(r'' + x_label)
    plt.ylabel(r'' + y_label)
    plt.grid(True)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image


def animate_cars(number_frame: int, lead_position_profile: List[float], acc_position_profile: List[float],
                 lead_acceleration_profile: List[float], acc_acceleration_profile: List[float],
                 lead_vehicle_param: Dict, acc_vehicle_param: Dict):
    """
    Creation of GIF with ACC vehicle following leading vehicle

    :param number_frame: current time step
    :param lead_position_profile: position profile of lead vehicle
    :param acc_position_profile: position profile of ACC vehicle
    :param lead_acceleration_profile: acceleration profile of lead vehicle
    :param acc_acceleration_profile: acceleration position profile of ACC vehicle
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    """
    if number_frame > len(lead_position_profile) - 1:
        number_frame = len(lead_position_profile) - 1

    x_position_lead = lead_position_profile[number_frame]
    x_position_acc = acc_position_profile[number_frame]
    position_differences = list(map(sub, lead_position_profile, acc_position_profile))
    max_position_difference = max(position_differences)
    acc_light_on = True
    lead_light_on = True
    color_green_acc = 0
    color_red_acc = 0.5
    color_blue_acc = 0
    color_green_lead = 0
    color_red_lead = 0.5
    color_blue_lead = 0

    mp.rcParams["figure.figsize"] = [20, 3]
    plt.cla()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(1, figsize=(5, 5), dpi=1200)
    ax = fig.add_subplot(111)

    x_min = x_position_lead - max_position_difference - 6
    x_max = x_position_lead + 6
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, 4)

    line_x = np.linspace(x_min, x_max, 1000)
    lower_line = np.zeros((len(line_x), 1))
    upper_line = np.ones((len(line_x), 1)) * 3
    ax.plot(line_x, upper_line, "k")
    ax.plot(line_x, lower_line, "k")

    # Create a Rectangle patch
    acc_vehicle = patches.Rectangle((x_position_acc - acc_vehicle_param.get("dynamics_param").l, 1.5
                                     - 0.5 * acc_vehicle_param.get("dynamics_param").w),
                                    acc_vehicle_param.get("dynamics_param").l,
                                    acc_vehicle_param.get("dynamics_param").w, linewidth=1, edgecolor='k',
                                    facecolor=(0.3, 0.3, 0.3, 1))

    lead_vehicle = patches.Rectangle((x_position_lead, 1.5
                                      - 0.5 * lead_vehicle_param.get("dynamics_param").w),
                                     lead_vehicle_param.get("dynamics_param").l,
                                     lead_vehicle_param.get("dynamics_param").w, linewidth=1, edgecolor='k',
                                     facecolor=(0.0, 0.0, 0.5, 0.75))

    # Set color of braking light
    if acc_acceleration_profile[number_frame] < -0.001:
        color_green_acc = 0
        color_red_acc = 0.5
        color_blue_acc = 0
    elif acc_acceleration_profile[number_frame] > 0.001:
        color_green_acc = 0.5
        color_red_acc = 0
        color_blue_acc = 0
    else:
        acc_light_on = False
    if lead_acceleration_profile[number_frame] < -0.001:
        color_green_lead = 0
        color_red_lead = 0.5
        color_blue_lead = 0
    elif lead_acceleration_profile[number_frame] > 0.001:
        color_green_lead = 0.5
        color_red_lead = 0
        color_blue_lead = 0
    else:
        lead_light_on = False

    # Draw braking light
    if acc_light_on:
        circle1 = plt.Circle((x_position_acc - acc_vehicle_param.get("dynamics_param").l, 1.5
                              - 0.5 * acc_vehicle_param.get("dynamics_param").w),
                             0.2, facecolor=(color_red_acc, color_green_acc, color_blue_acc))
        circle2 = plt.Circle((x_position_acc - acc_vehicle_param.get("dynamics_param").l, 1.5
                              + 0.5 * acc_vehicle_param.get("dynamics_param").w),
                             0.2, facecolor=(color_red_acc, color_green_acc, color_blue_acc))
        ax.add_artist(circle1)
        ax.add_artist(circle2)
    if lead_light_on:
        circle3 = plt.Circle((x_position_lead, 1.5
                              - 0.5 * lead_vehicle_param.get("dynamics_param").w), 0.2,
                             facecolor=(color_red_lead, color_green_lead, color_blue_lead))
        circle4 = plt.Circle((x_position_lead, 1.5
                              + 0.5 * lead_vehicle_param.get("dynamics_param").w), 0.2,
                             facecolor=(color_red_lead, color_green_lead, color_blue_lead))
        ax.add_artist(circle3)
        ax.add_artist(circle4)

    ax.add_patch(acc_vehicle)
    ax.add_patch(lead_vehicle)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image


def store_videos(node_list: List[Node], simulation_param: Dict, lead_vehicle_param: Dict, acc_vehicle_param: Dict):
    """
    Creation of profile videos and car following video

    :param node_list: list of nodes from the last time step
    :param simulation_param: dictionary with simulation specific parameters
    :param lead_vehicle_param: dictionary with physical parameters of the lead vehicle
    :param acc_vehicle_param: dictionary with physical parameters of the ACC vehicle
    """
    dt = simulation_param.get("dt")

    time, velocity_profile_lead, velocity_profile_acc, x_position_profile_lead, x_position_profile_acc, \
    acceleration_profile_lead, acceleration_profile_acc, x_distance_profile, safe_distance, unsafe_distance = \
        create_profiles(node_list, simulation_param, lead_vehicle_param, acc_vehicle_param)

    # Separate generation of each video is necessary
    # (otherwise it could be that the video is cut-off too early or the font sizes are not identical)
    imageio.mimsave('velocity.gif', [animate_profiles(i, velocity_profile_lead, velocity_profile_acc,
                                                      time, '$t~[s]$', '$v~[m/s]$', 0, max(time), 0 - 0.25,
                                                      max(lead_vehicle_param.get("dynamics_param").longitudinal.v_max,
                                                          acc_vehicle_param.get("dynamics_param").longitudinal.v_max)
                                                      + 0.25)
                                     for i in range(len(velocity_profile_lead) + 10)], fps=1 / dt)
    imageio.mimsave('acceleration.gif', [animate_profiles(i, acceleration_profile_lead, acceleration_profile_acc,
                                                          time, '$t~[s]$', '$a~[m/s^2]$', 0, max(time),
                                                          min(lead_vehicle_param.get("a_min"),
                                                              acc_vehicle_param.get("a_min")) - 0.25,
                                                          max(lead_vehicle_param.get("a_max"),
                                                              acc_vehicle_param.get("a_max")) + 0.25)
                                         for i in range(len(velocity_profile_lead) + 10)], fps=1 / dt)
    imageio.mimsave('position.gif', [animate_cars(i, x_position_profile_lead, x_position_profile_acc,
                                                  acceleration_profile_lead, acceleration_profile_acc,
                                                  lead_vehicle_param, acc_vehicle_param)
                                     for i in range(len(x_position_profile_lead) + 10)], fps=1 / dt)
