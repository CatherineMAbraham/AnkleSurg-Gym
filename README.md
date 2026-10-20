# AnkleSurg-Gym
AnkleSurg: Autonomous Deep Reinforcement Learning for Surgical Ankle Fracture Reduction

AnkleSurg is a high-fidelity PyBullet-based simulation framework designed for training and evaluating Deep Reinforcement Learning (DRL) agents in multi-DoF anatomical fracture reduction. The environment provides goal-conditioned state spaces, realistic soft-tissue and ligament surrogate modeling, and active force-bounded safety constraints for patient-specific orthopaedic intervention.

Key Features

6-DoF Kinematic Pose Reduction: Continuous control space for translational ($x, y, z$) and rotational ($\phi, \theta, \psi$) anatomical alignment.

Patient-Specific Anatomical Meshes: Direct support for patient URDF loading derived from clinical CT/X-ray dataset segmentation.

Biomechanical Soft-Tissue Surrogates: Integrated discrete spring meshes ($E = 1 \times 10^6 \text{ Pa}$) and volumetric VTK soft-tissue interaction models.

Safety & Force Constraints: Built-in velocity-barrier safety filter (safe_mode) and active contact-force thresholding to protect delicate anatomical structures (e.g., growth plates in Salter-Harris fractures).

Domain Randomization: Randomization modules for initial fracture pose, ligament stiffness/damping, spring counts, dynamics, and sensor noise.

Proven Policy Stability: Pre-configured and benchmarked using Twin Delayed Deep Deterministic Policy Gradient (TD3) for low-variance, deterministic surgical trajectory execution.

Installation & Prerequisites

Prerequisites

Python 3.8+

PyBullet

PyTorch

Gymnasium / OpenAI Gym

NumPy & SciPy

VTK (optional, for volumetric soft-tissue rendering)

Environment Setup

git clone https://github.com/your-org/AnkleSurg.git
cd AnkleSurg
pip install -r requirements.txt


Environment Configuration & Defaults

The core environment can be initialized with custom parameters or default configurations:

import gym
import anklesurg_env

env = gym.make('AnkleSurg-v0', 
    patient=102,                  # Patient URDF dataset index
    action_type='euler',          # 6-DoF Pose action parameterisation
    softtissue='spring',          # Discrete spring surrogate model
    safe_mode=False,              # Active velocity-barrier filter
    maxforce=3.5,                 # Maximum force threshold (N)
    maximum_contact_force_threshold=0.2 # Critical force cutoff (N)
)


Complete Argument Reference

Argument

Type

Default

Description

patient

int

None

Patient dataset ID for loading specific URDF bone meshes.

test

bool

False

Mode toggle (False: training mode, True: evaluation mode).

render_mode

str

None

PyBullet visualization ('human' or None).

vtk_file

str

None

Path to volumetric soft-tissue mesh file.

obs_type

str

'dict'

State observation vector layout ('dict' or 'flat').

action_type

str

'euler'

Kinematic control representation ('euler', 'pos_only', etc.).

dt

float

0.001

Action scaling magnitude for spatial steps ($1.0\text{ mm}$).

dr

float

0.1

Action scaling magnitude for angular steps ($0.1\text{ rad}$).

start_pos

str

'home'

Initial manipulator configuration ('home' or 'extended').

goal_type

str/list

'random'

Target pose setup ('random' or fixed 6-element pose vector).

distance_threshold_pos

float

0.0005

Position convergence tolerance ($0.5\text{ mm}$).

distance_threshold_ori

float

0.0087

Orientation convergence tolerance ($0.5^\circ \approx 0.0087\text{ rad}$).

max_steps

int

100

Maximum episode step limit.

horizon

str

'variable'

Episode termination strategy ('variable' or 'fixed').

reward_type

str

'sparse'

Reward formulation ('sparse' or 'dense').

softtissue

str

'spring'

Soft-tissue model formulation ('spring', 'soft', or 'none').

youngs_modulus

float

1e6

Elastic modulus ($E = 1 \times 10^6\text{ Pa}$) of tissue surrogate.

youngs_modulus_type

str

'None'

Parameter assignment regime ('None', 'eval_mode', 'testing').

number_of_springs

int

3

Number of discrete spring elements instantiated.

width

float

0.005

Structural width ($5.0\text{ mm}$) of discrete spring mesh.

contact_type

int

0

Contact interaction mode ($0$: disabled, $1$: force penalty active).

maxforce

float

3.5

Operational interaction force limit ($3.5\text{ N}$).

maximum_contact_force_threshold

float

0.2

Emergency contact force limit ($0.2\text{ N}$) during testing.

safe_mode

bool

False

Toggles deterministic velocity-barrier safety filter.

randomise_start

bool

False

Enables initial distal bone pose perturbations.

randomise_ligs

bool

False

Enables stochastic variation of ligament stiffness/damping.

randomise_num_springs

bool

False

Randomizes active spring count in surrogate soft tissue.

randomise_foot_dynamics

bool

False

Randomizes bone mass, inertia, and surface friction.

randomise_sensor_noise

bool

False

Injects zero-mean Gaussian noise into state feedback.

Basic Usage Example

Below is a minimal script demonstrating an episode execution loop with random actions:

import gym
import numpy as np
import anklesurg_env

env = gym.make('AnkleSurg-v0', render_mode='human', patient=102)
obs = env.reset()

done = False
total_reward = 0.0

while not done:
    # Sample action within action space bounds
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    if info.get('force_violation', False):
        print(f"Safety constraint violated! Peak force: {info['peak_force']:.3f} N")

print(f"Episode complete. Total Reward: {total_reward:.3f}")
