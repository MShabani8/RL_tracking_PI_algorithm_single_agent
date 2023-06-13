# RL_tracking_PI_algorithm_single_agent
the code implements a training algorithm for a tracking control system using dynamic programming and reinforcement learning. It uses neural networks to approximate the control policy and iteratively updates the networks to improve the system's performance.

The code begins by setting up the necessary variables and parameters for the control system.

It defines the action network (actor) and the critic network (critic) using neural networks. These networks will be used to approximate the optimal control policy.

The code initializes the state and control matrices and sets the values for the performance index, as well as other parameters such as the learning rates and training epochs for the actor and critic networks.

It generates training data by randomly creating different states for the reference system (x_tr) and the agent system (x_ta). The training data is stored in the variable e_train.

The code enters a loop where it trains the critic and actor networks iteratively. In each iteration, it updates the critic network by evaluating the policy using the evaluate_policy function and then trains the critic network using the training data and the evaluated critic target.

The code also calculates and stores the performance index for each iteration, which represents the performance of the control system.

After the training loop is completed, the code plots the performance index over iterations and saves the trained actor and critic networks.

The evaluate_policy function evaluates the policy defined by the actor network by simulating the control system for a given number of steps (eval_step) using the provided initial state (e) and reference and agent system states (x_tr and x_ta). It returns the sum of the cost function values obtained during the simulation.

The controlled_system function calculates the output of the control system for a given agent state, reference state, and control input. It uses the system matrices A and B.

The cost_function function calculates the cost function value for a given error (e) and control input (u). It uses the weight matrices Q and R.
