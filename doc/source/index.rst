.. ACC Falsification Tool documentation master file, created by
   sphinx-quickstart on Wed Oct  2 08:41:58 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ACC Falsification Tool Documentation
==================================================

The Falsification tool for Adaptive Cruise Control (ACC) systems provides an interface to test ACC systems.
The tool is based on the approaches introduced in
`M. Koschi, C. Pek, S. Maierhofer, and M. Althoff: Computationally Efficient Safety Falsification of Adaptive Cruise Control Systems, in Proc. of the IEEE Int. Conf. on Intelligent Transportation Systems, 2019 <https://mediatum.ub.tum.de/doc/1514806/621691597458.pdf>`_.

Prerequisites
=============

For the execution of the falsification tool you need at least Python 3.6
and the following modules:

* commonroad-io >= 2019.2
* imageio >= 2.5.0
* matplotlib >= 3.1.0
* numpy >= 1.16.4
* ruamel.yaml >= 0.16.5
* scipy >= 1.3.1

| The usage of the Anaconda Python distribution is recommended.
| You can install the required Python packages with the provided requirements.txt file (*pip install -r requirements.txt*).
| Additionally, you need the `CommonRoad vehicle models <https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/tree/master/Python>`_ which must be added to your Python interpreter path.



Getting Started
===============
The three main files are *falsification.py*, *config.yaml*, and *review.py*:

**1. config.yaml:** This file allows to adapt all parameters used by
the falsification system, e.g. ACC vehicle parameters, simulation parameters, leading vehicle parameters).

**2. falsification.py:** This file starts the falsification, e.g. by executing it within the command line ("python falsification.py > solution.out").
If activated in config.yaml or if a valid solution is found, solution files are generated with
the initial setup (config_xyz.yaml) and the lead vehicle trajectory (trajectory_xyz.pkl), where xyz in the name of the
generated files corresponds to a concatenation of the current date and time.
The initial state of the ACC vehicle corresponds to the values defined in config.yaml (*s_init*, *v_init*).
The initial state of the leading vehicle is randomly set in front of the ACC vehicle according to the parameters
in config.yaml (*s_safe_init_backward*, *delta_s_init_min*, *delta_s_init_max*).

**3. review.py:** This file allows to execute a stored initial setup and leading vehicle
trajectory.
The script plots different profiles of the leading and ACC vehicle and stores a CommonRoad scenario.
You must provide an input parameter to the script (e.g., "python review.py xyz",
where the solution files are called config_xzy.yaml and trajectory_xyz.pkl).

The folder **acc** contains different ACC systems.
If you want to add your own ACC system,
you must implement the ACC system corresponding to the interface
in **acc_interface.py**. Additionally, you must add your ACC system to the dictionaries in **config.yaml**
and to the create_acc_vehicle_param() function and the ACCController enum, both located in **configuration.py**.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contact information
===================

:Website: `https://gitlab.lrz.de/tum-cps/safety-falsification-acc <https://gitlab.lrz.de/tum-cps/safety-falsification-acc>`_
:Email: `sebastian.maierhofer@tum.de <sebastian.maierhofer@tum.de>`_