# AnkleSurg: Autonomous DRL Fracture Reduction Framework

AnkleSurg is a PyBullet-based Deep Reinforcement Learning (DRL) environment designed for autonomous, force-constrained orthopaedic fracture reduction. The framework integrates patient-specific bone anatomy (URDF/VTK meshes), discrete and continuous biomechanical soft-tissue surrogates, safety barrier filtering, and automated goal-pose targeting for intra-articular ankle fracture alignment.

---

## Key Features

- **Biomechanical Soft-Tissue Surrogates:** Supports spring surrogates, volumetric deformable bodies, and configurable ligament dynamics.
- **Safety-Critical Force Constraints:** Integrated contact-force monitoring, velocity-barrier safety filtering (`safe_mode`), and adjustable maximum force thresholds.
- **Patient-Specific Datasets:** Native loading of patient bone meshes (e.g., Patients 102, 132, 198, 252) with custom coordinate frames.
- **Domain Randomization:** Configurable randomizations for distal fragment start poses, ligament stiffness/damping, spring count, sensor noise, and bone inertial/frictional dynamics.
- **Algorithms & Evaluation:** Standardized interface compatible with off-policy continuous control algorithms (TD3, SAC).

---

## Default Environment Parameters

The environment defaults and configuration options are summarized below:

| Parameter | Type | Default | Description / Units |
| :--- | :--- | :--- | :--- |
| `patient` | `int` | `None` | Patient ID index for loading URDF bone meshes |
| `test` | `bool` | `False` | Execution mode (`False`: training, `True`: testing/evaluation) |
| `render_mode` | `str` | `None` | PyBullet visualization mode (`'human'` or `None`) |
| `vtk_file` | `str` | `None` | Path to volumetric soft-tissue mesh file |
| `obs_type` | `str` | `'dict'` | State observation vector formulation (`'dict'` or `'flat'`) |
| `action_type` | `str` | `'euler'` | Action space parameterization (`'euler'`, `'pos_only'`, etc.) |
| `dt` | `float` | `0.001` | Action translation step magnitude ($1.0\text{ mm}$) |
| `dr` | `float` | `0.1` | Action rotation step magnitude ($0.1\text{ rad}$) |
| `start_pos` | `str` | `'home'` | Initial manipulator pose configuration (`'home'` or `'extended'`) |
| `goal_type` | `str`/`list` | `'random'` | Target alignment pose setup (`'random'` or 6-DoF pose vector) |
| `distance_threshold_pos` | `float` | `0.0005` | Positional goal convergence tolerance ($0.5\text{ mm}$) |
| `distance_threshold_ori` | `float` | `0.0087` | Rotational goal convergence tolerance ($0.5^\circ \approx 0.0087\text{ rad}$) |
| `max_steps` | `int` | `100` | Maximum step limit per episode before termination |
| `horizon` | `str` | `'variable'` | Episode termination mode (`'variable'` on target hit, or `'fixed'`) |
| `reward_type` | `str` | `'sparse'` | Scalar reward signal formulation (`'sparse'` or `'dense'`) |
| `softtissue` | `str` | `'spring'` | Soft tissue formulation (`'spring'`, `'soft'`, or `'none'`) |
| `youngs_modulus` | `float` | `1e6` | Elastic modulus ($E = 1\times10^6\text{ Pa}$) of tissue structure |
| `youngs_modulus_type` | `str` | `'None'` | Parameter assignment regime (`'None'`, `'eval_mode'`, `'testing'`) |
| `number_of_springs` | `int` | `3` | Number of discrete spring elements in surrogate model |
| `width` | `float` | `0.005` | Structural width ($5.0\text{ mm}$) of spring tissue mesh |
| `contact_type` | `int` | `0` | Contact interaction mode ($0$: disabled, $1$: force penalty active) |
| `maxforce` | `float` | `3.5` | Operational force threshold ($3.5\text{ N}$) |
| `maximum_contact_force_threshold` | `float` | `0.2` | Emergency contact force limit ($0.2\text{ N}$) triggering test termination |
| `safe_mode` | `bool` | `False` | Toggles active execution of velocity-barrier safety filter |
| `randomise_start` | `bool` | `False` | Enables initial distal bone fragment pose perturbations |
| `randomise_ligs` | `bool` | `False` | Randomizes ligament stiffness and damping properties |
| `randomise_num_springs` | `bool` | `False` | Randomizes active spring count in soft-tissue surrogate |
| `randomise_foot_dynamics` | `bool` | `False` | Randomizes bone inertial properties and friction coefficients |
| `randomise_sensor_noise` | `bool` | `False` | Injects zero-mean Gaussian noise into state feedback |

---

## Installation
