### Edge Computing and Caching Optimization based on PPO for Task Offloading in RSU-assisted IoV

This is the runnable Python project for the paper 'Edge Computing and Caching Optimization based on PPO for Task Offloading in RSU-assisted IoV'.

#### Guidance

To run the project, run the .py files in './run_this'.

'run_object.py' generates the results of the main objects.

'run_different_capacity.py' generates the comparison between different cache capacities.

'run_different_deadline.py' generates the comparison between different task deadlines.

'run_different_algo1.py' generates the comparison between our scheme and the LFU and LRU algorithms.

'run_different_algo2.py' generates the comparison between our scheme and the optim caching scheme.

Note: only run one py file at a time.

#### Components

'./envs' contains environment settings, parameter configurations and the data struct of the project.

'./methods' contains the discrete PPO algorithm.

'./run_this/data' contains the stored data of the project.

'./run_this/run_methods_and_outputs' contains the execution function of runnable files.

'./run_this/run_methods_and_outputs/outputs' contains the outputs of runnable files.



@author: Mason Lynn