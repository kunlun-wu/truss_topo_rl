# Reinforcement Learning for Truss Topology Optimization

## 1. Problem Statement
Truss topology optimization involves determining the optimal arrangement of truss members to achieve desired structural performance while minimizing material usage and cost. This project aims to leverage reinforcement learning (RL) techniques to develop an efficient algorithm for truss topology optimization. This is a multi-objective optimization problem with goals:
1. Minimize material usage (weight/cost)
2. Maximize structural performance (stiffness, strength)

I hope for this agent to navigate the combinational landscape of discrete search space (binary add/remove members, for a grid with N possible connections, the search space is 2^N) and discontinuous reward function (small changes in topology can lead to large changes in performance), and ultimately discover efficient structural forms, such as triangulation or arches, without being explicitly programmed with engineering rules.

(Like the Cook's Membrane project, I also encountered many challenges in this project. I still carry the PTSD from the volumetric locking I experienced there. I realized the distinct difficulty of a priori machine learning compared to the more traditional posteriori supervised learning. Despite the challenges, I am still really interested in the a priori approach, so I did this project to further explore it.)

## 2. Stack and Design
#### Physics Engine: ```anastruct```
```anastruct``` is a lightweight Python library for structural analysis of 2D and 3D truss and frame structures. It allows for rapid iteration within the Python loop, suitable for RL training which requires numerous simulation steps. It provides immediate feedback on global stability, calculating stiffness matrices to determine if a structure is statically determinate or unstable.
#### RL Framework: ```Gymnasium & Tensorflow```
Environment(```TrussEnv```) is a custom Gymnasium environment that discretizes the design domain into a grid of nodes. The state space is a binary vector representing the existence of every possible bar connection in the grid, and the action space is a discrete action space where the agent selects a bar index to toggle, this allows the agent to iteratively refine the structure. 

Algorithm is a Deep Q-Network (DQN) implemented using TensorFlow. Since the action space is discrete, DQN is a good fit. I used Experience Replay to break correlations between consecutive design steps and a Target Network to stabilize the learning of Q-values. The reward function is designed to balance material usage and structural performance, with penalties for unstable configurations.
#### Reward Function
The project uses the initial grid in a state where all possible bars present, then the agent tries to remove unnecessary bars. The reward function was R = C_base - (w * Mass) - (k * Displacement). If the solver detects an unstable structure, the agent receives a harsh penalty and the episode terminates (this can be adjusted to allow further exploration for harder configurations). This encourages the agent to find stable structures. By tuning the weights w and k, I can prioritize lighter structures or stiffer structures.
### File Structure
```main.py``` - Entry point to train and evaluate the RL agent. Ask for configuration file and define the training duration. 
```train.py``` - Manages the training loop. Initializes environment and agent, runs episode loops, and manages experience replay. It logs metrics, saves model checkpoints, and generates snapshots of the best structures found.
```evaluate.py``` - Visualizes results and benchmarking performance. Plots the training curve, reconstructs and saves images of the best and final truss designs, runs a benchmark comparing the trained RL agent against a random search baseline.
```environment.py``` - The bridge between the RL agent and the physics engine. Defines state space, action space, and reward function.
```agent.py``` - The brain of the system. Implements DQN using TensorFlow, ```act``` uses an epsilon-greedy strategy to balance exploration and exploitation (using best guesses). ```replay``` samples minibatches of past experiences to train the Q-network, breaking correlations for stable learning. ```update_target_model``` maintains a separate target network to stabilize Q-value targets.
```physics.py``` - The simulation engine . Initializes grid, generating nodes and potential connections based on config, cleans (```_get_cleaned_structure```) disconnected bars and dangling nodes to ensure solver does not crash on trivial errors, and solves using ```anastruct``` library to apply supports and loads, returning displacement and total weight.

## 3. Challenges and Iterations
#### 3.1 Support Constraints (All fixed supports vs combination of fixed and rolling supports)
Observation: initial attempts had missing bars even in simple configs that appeared unstable and unphysical. 

Realization: I used the default fixed supports for all supports in modeling, which allowed distribution of force in unrealistic/counter-intuitive ways.

Solution: By imitating real-world constraints of bridges (rolling support in some places to allow thermal expansion), this problem was resolved.
#### 3.2 Welded vs Hinged Joints/Nodes
Observation: Initial attempts produced weird structures with many "floating" members that did not connect to anything, sometimes the agent favored rectangular, frame-like structures.

Realization: I used the default welded joints in modeling. which enabled structures to rely on the bending strength of joints rather than triangulation.

Solution: By switching all joints to hinged, this problem was resolved.
#### 3.3 "house of cards" instability
Observation: The agent would sometimes produce "house of cards" structures that were theoretically stable in the solver but would collapse in reality.

Realization: The solver only checks for gravity, which is too insufficient. The agent was overfitting to the vertical vector and ignoring lateral stability.

Solution: I added a lateral "wind load" (20% of vertical load) to the physics simulation, which forced the agent to consider lateral stability.
#### 3.4 RL stability
Observation: During training, the loss function would oscillate wildly, and the agent would forget previously learned good strategies.

Realization: The netword updates its weight using its own predictions as the target labels, this creates a feedback loop where the target moves with every update, leading to instability.

Solution: I implemented a Target Network. It is a secondary, frozen copy of the NN that is used to calculate stable target values. It is only updated/synced periodically, providing a stable anchor for the learning process.
#### 3.5 Hardcoded vs Dynamic Penalties
Observation: When switching to another config file, the agent struggled to learn.

Realization: The hardcoded penalties were brittle, the magnitude of the displacement reward varies depending on the grid size, if the penalty is too low, the agent ignores safety, if it was too high, the agent is terrified of taking any action.

Solution: I shifted to dynamic penalties from the config files.
#### 3.6 Visualization of Progress
Observation: I saw numbers improving in the logs but had no idea what the agent was actually building until the very end. This makes the iteration cycle very inefficient.

Solution: I implemented dynamic snapshot saving, whenever the agent find a new highest scoring structure, the system renders the structure and saves an image. This allows me to see the agent's progress over time.
## 4. Results and Discussion
I ran four configurations, the files are located in ```configs/```. I am summarizing the results below (red triangles indicate fixed supports, red triangles with a line indicates rolling supports, red arrow indicates applied load, the orange transparent rectangles indicate tension (positive force) and the blue transparent rectangles indicate compression (negative force)):
#### 4.1 very simple bridge ```default_config.json```
This config was used to build and test the project as a starting point. It is a very simple bridge with a 2x3 grid and the load applied in the bottom center. The left support is a fixed support while the right support is a rolling support for thermal expansion. The agent quickly learned to remove unnecessary bars and form a simple triangular truss structure. The final design was efficient, using minimal material while maintaining stability.

Snapshots:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/snapshots/snapshot1.png">
    </a>
</p>
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/snapshots/snapshot2.png">
    </a>
</p>
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/snapshots/snapshot3.png">
    </a>
</p>

We can see the agent is learning to remove the top corners and form a triangular arch structure. although triangulation or arch is somewhat of an overstatement for this simple case.

Learning Curve:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/plot_learning_curve.png">
    </a>
</p>

Best Model Structure:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/result_best.png">
    </a>
</p>

Final Model Structure:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/result_final.png">
    </a>
</p>

The best model structure and the final model structure are exactly the same, which means stable convergence, the agent has shifted from exploring random bars to exploiting the optimal design, it found a peak in teh reward landscape and stayed there.

Evaluations:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/eval_1_score_dist.png">
    </a>
</p>

For the score distribution, we can see the agent predicted model is top in the valid structures compared to random search, indicating the agent has learned a good policy for this configuration (the large gray bar on the left are invalid structures that collapsed).

<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/eval_2_success_rate.png">
    </a>
</p>
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/default_results/eval_3_design_space.png">
    </a>
</p>

#### 4.2 preparation for discovering arch ```variation_config.json```
This config is a further proof of concept simple file to test the agent while preparing for the next more complex config. It features the same 2x3 grid but with the supports at the left corners and the load at the top-right corner, essentially mimicking half of a arched bridge.

Learning Curve:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_results/plot_learning_curve.png">
    </a>
</p>

Best Model Structure:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_results/result_best.png">
    </a>
</p>

Final Model Structure:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_results/result_final.png">
    </a>
</p>

The structures are different but similar, both are valid. It might required more training time to fully converge.

Evaluations:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_results/eval_1_score_dist.png">
    </a>
</p>

For the score distribution, we can see the agent predicted model is close to top in the valid structures compared to random search, indicating the agent has learned a good policy, but might not converged fully.

<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_results/eval_2_success_rate.png">
    </a>
</p>
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_results/eval_3_design_space.png">
    </a>
</p>

#### 4.3 discovering arch structure ```variation_hard_config.json```
This config features a 3x5 grid with fixed supports at the bottom corners and the load at the top center, with rolling supports on the top corners to allow for thermal expansion, mimicking a full arched bridge. I trained using this config hoping for the agent to discover the arch shape to efficiently transfer loads to the supports.

Learning Curve:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_hard_results/plot_learning_curve.png">
    </a>
</p>

Best Model Structure:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_hard_results/result_best.png">
    </a>
</p>

Final Model Structure:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_hard_results/result_final.png">
    </a>
</p>

The best model structure and the final model structure are very different, and the best model structure is the original grid with no bars removed. This indicates 1. that the agent has not fully converged after the 2000 episodes of training duration I set, and 2. that the penalty weight for displacement compared to the penalty weight for material cost is too high, causing the agent to focus on stability over material efficiency.

However, in the snapshots, we can see that the agent is capable of learning arch-like structures with triangulation and symmetry. Given more time and better penalty weights/hyperparameter tuning, the model would likely converge to an efficient arch structure.

Snapshots:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_hard_results/snapshots/snapshot3.png">
    </a>
</p>
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_hard_results/snapshots/snapshot4.png">
    </a>
</p>

Evaluations:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_hard_results/eval_1_score_dist.png">
    </a>
</p>

For the score distribution, we can see the agent predicted model is scoring on the lower end of the valid structures from random search. The reason is as discussed above.

<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_hard_results/eval_2_success_rate.png">
    </a>
</p>
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/variation_hard_results/eval_3_design_space.png">
    </a>
</p>

#### 4.4 stress testing larger grid ```stressTest_config.json```
This config is a stress test to evaluate the scalability of the RL approach. It features a larger 3x7 grid with the same boundary conditions of the first default config. The increased grid size significantly expands the design space, challenging the agent to find efficient structures within a more complex environment.

I did not have time to fully train using the stress test config, but I have snapshots of the intermediate best structures:

Snapshots:
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/stressTest_results/snapshots/snapshot1.png">
    </a>
</p>
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/stressTest_results/snapshots/snapshot2.png">
    </a>
</p>
<p align="center">
    <a href="https://github.com/kunlun-wu/truss_topo_rl">
        <img src="https://raw.githubusercontent.com/kunlun-wu/truss_topo_rl/main/results/stressTest_results/snapshots/snapshot3.png">
    </a>
</p>

We can see that the agent is learning to form an arch structure with triangulation and symmetry, snapshot 2 shows a resemblance to the Baltimore truss design, and snapshot 3 shows a resemblance to the Pratt/Howe truss design, both being well-known efficient truss configurations. Given more training time, I believe the agent could further refine these designs to resemble classic structures.

## 5. Potential Improvements and Reflections
While the current implementation demonstrates the feasibility of using reinforcement learning for truss topology optimization, there are several avenues for improvement and future work:
- Graph Neural Networks (GNNs): Currently, the truss is represented as a flat vector, losing the spatial "neighborhood" information of the grid. A GNN could represent nodes and bars directly as a graph. This would allow the trained model to generalize to grids of different sizes (scale-invariance), which the current fixed-input Dense network cannot do. 
- Symmetry Constraints: Real-world civil engineering structures are almost always symmetric. Enforcing symmetry in the action space (e.g., if the agent adds a bar on the left, the environment automatically adds the mirrored bar on the right) would cut the search space in half and guarantee aesthetically pleasing results.
- Curriculum Learning: For large grids, instead of training on a difficult grid immediately, we could implement a curriculum where the agent first masters a 3x3 grid. The weights could then be transferred (or the grid expanded) to progressively harder tasks.
- Continuous Node Optimization: Currently, nodes are locked to a grid. A hybrid approach could use RL for topology (connectivity) and a gradient-based optimizer (like scipy.optimize) to fine-tune the exact x,y coordinates of the nodes, allowing for more organic, highly efficient shapes.

Currently, my agent suffers from very narrow limitations: a model trained on a 3x5 arch bridge cannot solve a 3x5 cantilever beam because it has memorized the spatial location of the supports rather than understanding the physics of supports. Furthermore, changing the grid size from 3x5 to 3x6 breaks the neural network entirely because the input layer size changes. Imagining the evolution of this project into a general-use tool, I think of a two-stage roadmap:
1. Transfer Learning:

    Instead of initializing every training run with random noise, I can warm-start new problems using pre-trained weights. A model that has learned to build triangles and connect loads to supports in a "Standard Bridge" task already understands basic stability. If I load these weights into a "Cantilever" task, the agent starts with a fundamental understanding of connectivity, drastically reducing training time.

    I can create a library of models trained on simple tasks (vertical load, shear load). When a user inputs a complex custom task, the system loads the closest matching primitive models as the starting point for fine-tuning.

2. Universal Truss Solver (Context-Aware Inputs)

    For the ideal end-goal of this project, a single model capable of solving any boundary condition without retraining, I must change what the agent "sees."

    Right now, the agent sees only the structure (Bar_1_Exists, Bar_2_Exists...). It is blind to the environment; it doesn't know where the supports are, it only learns them by trial-and-error. To change this, I would feed the problem definition into the network alongside the structure. The input would become a multi-channel tensor (Channel 1: Structure of existing bars, Channel 2: Support locations, Channel 3: Load map represented as a vector fields of forces).

    By seeing the supports as part of the input, the agent can learn the causal relationship like always connecting the load path to supports. Then the agent could instantly solve a bridge with supports in random locations, effectively becoming a generalized structural engineer.