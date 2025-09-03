## Modules to Import
import gymnasium as gym
from gymnasium import spaces
import os
import pybullet as p
import pybullet_data
import numpy as np
from gym_fracture.envs.utils import calculate_distances, make_scene, getStarts, getGoal, check_done, get_new_pose, unpack_action
class fracturesurgery_env(gym.Env):
    def __init__(
        self,
        render_mode:str =None,
        reward_type:str ='sparse',
        distance_threshold_pos:float =0.005,
        distance_threshold_ori:float =0.05,
        max_steps:int =50,
        obs_type:str ='dict',
        goal_type:str ='random',
        dv:float =0.05,
        action_type:str ='sixactions',
        horizon:str ='variable',
    ):
        """An environment for training a RL agent to perform ankle fracture reduction surgery. 

           Args:
               render_mode (str): The mode for rendering the environment (e.g., 'human' or 'None').
               reward_type (str): The type of reward to use ('sparse' or 'dense').
               distance_threshold_pos (float): The position distance threshold for success.
               distance_threshold_ori (float): The orientation distance threshold for success.
               max_steps (int): The maximum number of steps per episode.
               obs_type (str): The type of observation space ('dict' or 'Flat').
               goal_type (str): The type of goal ('random' or 'fixed').
               dv (float): The action multiplier for the action space.
               action_type (str): The type of action space ('sixactions', 'fouractions', 'pos_only', or 'ori_only').
               horizon (str): The horizon type ('fixed' or 'variable').
        """
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.goal_type = goal_type
        self.reward_type = reward_type
        self.dv = dv
        self.max_steps = max_steps
        self.action_type = action_type
        self.horizon = horizon
        self.success_threshold = 0.6
        self.episodes_done = 0

        if self.render_mode == 'human':
            p.connect(p.GUI, options="--background_color_red=0.9686--background_color_blue=0.79216--background_color_green=0.7882")
        else:
            p.connect(p.DIRECT)
        self.connected = True
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

        # Observation space
        
        if self.action_type == 'ori_only':
            obs_shape = 30  # Reduced from 31
            goal_shape = 5
        elif self.action_type == "pos_only":
            obs_shape = 30  # Reduced from 31
            goal_shape = 4
        else:
            obs_shape = 31
            goal_shape = 8
        if self.obs_type == 'dict':
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=-200, high=200, shape=(obs_shape,), dtype=np.float32),
                'achieved_goal': spaces.Box(low=-200, high=200, shape=(goal_shape,), dtype=np.float32),
                'desired_goal': spaces.Box(low=-200, high=200, shape=(goal_shape,), dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(low=-200, high=200, shape=(obs_shape,), dtype=np.float32)

        # Action space
        self._set_action_space()

        self.goal_range_low = np.zeros(3)
        self.goal_range_high = np.zeros(3)
        self.goal_ori_low = np.zeros(3)
        self.goal_ori_high = np.zeros(3)
        self.current_step = 0
        self.pos_distance = 0.0
        self.angle = 0.0
        self.distance_threshold_pos = distance_threshold_pos
        self.distance_threshold_ori = distance_threshold_ori
        self.pitch = 0.0
        self.n = 0

    def _set_action_space(self):
        if self.action_type == 'quat':
            self.action_space = spaces.Box(
                low=np.array([-0.005, -0.005, -0.005, -0.05, -0.05, -0.05, 0.9]),
                high=np.array([0.005, 0.005, 0.005, 0.03, 0.03, 0.03, 1]),
                shape=(7,)
            )
        elif self.action_type == 'joint':
            mean_diff = np.array([
                5.344368461085916e-06, -1.7254826772920976e-06, -5.652514741346842e-06,
                -1.7459863701878556e-06, 3.352876319352558e-05, 2.7700700928288767e-06,
                -3.7586469804521616e-05, -1.1339311907951961e-06, 1.4051723622599417e-06
            ])
            low = mean_diff - 0.005
            high = mean_diff + 0.005
            self.action_space = spaces.Box(low=low, high=high, shape=(9,))
        elif self.action_type == 'ori_only':
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
            # low=np.array([-0.0007, -0.0013, -0.0014]),
              #  high=np.array([0.001, 0.0015, 0.0014]),
        elif self.action_type == 'pos_only':
            self.action_space = spaces.Box(low =-1, high=1, shape=(3,))
                #low=np.array([-0.005, -0.005, -0.005]),
                #high=np.array([0.005, 0.005, 0.005]),
                #shape=(3,)
            #)
        elif self.action_type == 'fiveactions':
            self.action_space = spaces.Box(low=-1, high=1, shape=(5,))
        elif self.action_type == 'fouractions':
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(6,))

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == 'sparse':
            # Handle ori_only case
            if self.action_type == 'ori_only':
                if achieved_goal.ndim == 1:
                    new_ori = achieved_goal[:4]
                    goal_ori = desired_goal[:4]
                    self.isHolding = achieved_goal[4]
                    dot_product = np.clip(np.abs(np.sum(new_ori * goal_ori)), -1.0, 1.0)
                    self.angle = 2 * np.arccos(dot_product)
                    reward = 0 if (self.angle <= self.distance_threshold_ori and self.isHolding == 1) else -1
                else:
                    new_ori = achieved_goal[:, :4]
                    goal_ori = desired_goal[:, :4]
                    self.isHolding = achieved_goal[:, 4]
                    dot_product = np.clip(np.abs(np.sum(new_ori * goal_ori, axis=-1)), -1.0, 1.0)
                    self.angle = 2 * np.arccos(dot_product)
                    reward = np.where(
                        (self.angle <= self.distance_threshold_ori) & (self.isHolding == 1),
                        0, -1
                    )
                return np.array(reward)

            # Handle pos_only case
            if self.action_type == 'pos_only':
                if achieved_goal.ndim == 1:
                    self.pos_distance = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
                    self.isHolding = achieved_goal[3]
                    reward = 0 if (self.pos_distance <= self.distance_threshold_pos and self.isHolding == 1) else -1
                else:
                    pos_achieved = achieved_goal[:, :3]
                    pos_desired = desired_goal[:, :3]
                    self.isHolding = achieved_goal[:, 3]
                    self.pos_distance = np.linalg.norm(pos_achieved - pos_desired, axis=1)
                    reward = np.where(
                        (self.pos_distance <= self.distance_threshold_pos) & (self.isHolding == 1),
                        0, -1
                    )
                return np.array(reward)

            # Handle general case (position + orientation)
            if achieved_goal.ndim == 1:
                pos_achieved, angle_achieved = achieved_goal[:3], achieved_goal[3:7]
                pos_desired, angle_desired = desired_goal[:3], desired_goal[3:7]
                self.pos_distance, self.angle = calculate_distances(self, pos_achieved, angle_achieved, pos_desired, angle_desired)
                self.isHolding = achieved_goal[7]
                reward = 0 if (
                    self.pos_distance <= self.distance_threshold_pos and
                    self.angle <= self.distance_threshold_ori and
                    self.isHolding == 1
                ) else -1
            else:
                pos_achieved, angle_achieved = achieved_goal[:, :3], achieved_goal[:, 3:7]
                pos_desired, angle_desired = desired_goal[:, :3], desired_goal[:, 3:7]
                self.pos_distance, self.angle = calculate_distances(self, pos_achieved, angle_achieved, pos_desired, angle_desired)
                self.isHolding = achieved_goal[:, 7]
                reward = np.where(
                    (self.pos_distance <= self.distance_threshold_pos) &
                    (self.angle <= self.distance_threshold_ori) &
                    (self.isHolding == 1),
                    0, -1
                )
            return np.array(reward)

        elif self.reward_type != 'sparse':
            d1 = self.pos_distance + self.angle
            d2 = self.pos_distance + self.angle
            d_pos = np.float32(self.pos_distance)
            rewardDistance = np.exp(-0.1 * self.pos_distance)
            rewardOrientation = np.exp(-0.1 * self.angle)
            e = rewardDistance + rewardOrientation

            if self.reward_type == 'dense' and self.action_type == 'pos_only':
                return -d_pos
            elif self.reward_type == 'dense_1' and self.horizon == 'variable':
                return -d1
            elif self.reward_type == 'dense_1' and self.horizon == 'fixed':
                return -d1 + e
            
       

    def reset(self, seed=None, options=None):
        self.n += 1
        p.resetSimulation()
        self.current_step = 0
        make_scene(self)
        fracturestart, fractureorientationDeg, legstartpos = getStarts(self)
        if isinstance(self.goal_type, str):
            getGoal(self, fracturestart, fractureorientationDeg)
        else:
            self.goal_pos = np.array(fracturestart.copy())
            self.goal_ori = np.array(self.goal_type)
            self.goal_range_low = fracturestart - [0.0125, 0.01, 0.003]
            self.goal_range_high = fracturestart + [0.0125, 0.02, 0.003]
            self.goal_ori_low = np.radians(fractureorientationDeg - [15, 5, 15])
            self.goal_ori_high = np.radians(fractureorientationDeg + [15, 5, 15])

        currentDir = os.path.dirname(os.path.abspath(__file__))
        leg_path = os.path.join(currentDir, "Assets/leg.urdf")
        foot_path = os.path.join(currentDir, "Assets/foot.urdf")
        self.objectUid = p.loadURDF(foot_path, basePosition=fracturestart, globalScaling=1)
        p.changeDynamics(self.objectUid, -1, mass=0.6, lateralFriction=0.5)
        self.leg = p.loadURDF(leg_path,
                        basePosition =legstartpos,
                        #baseOrientation = legorientation,
                        globalScaling = 1,
                        useFixedBase = 1)
        
        target_positions = np.array([0.00, 0.00])
        forces = [50, 50]
        for _ in range(50):
            p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, targetPosition=target_positions[0], force=forces[0])
            p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, targetPosition=target_positions[1], force=forces[1])
            p.stepSimulation()
            # time.sleep(1./500.)  # Remove for speed

        p.setGravity(0, 0, -9.8)
        for _ in range(10):
            p.stepSimulation()
            # time.sleep(0.002)  # Remove for speed

        self.target_position = np.concatenate((self.goal_pos, self.goal_ori))
        
        # Dummy visual shape for goal marker
        
        
        goal_cube = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=self.visual_shape,
                          basePosition=fracturestart, baseOrientation=self.goal_ori)

        
        initialpos = p.getLinkState(self.pandaUid, 11)[0]
        initialor = p.getLinkState(self.pandaUid, 11)[1]
        initialholdObject = len(p.getContactPoints(self.pandaUid, self.objectUid))
        initialisHolding = 1 if initialholdObject > 0 else 0
        initialvel = p.getLinkState(self.pandaUid, 11, 1)[6]
        initialJointPoses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
        initialJointVelocities = [p.getJointState(self.pandaUid, i)[1] for i in range(9)]
        self.pos_distance, self.angle = calculate_distances(self, initialpos, initialor, self.goal_pos, self.goal_ori)
        initialisHolding = int(initialisHolding)
        if self.action_type == 'ori_only':
            observation = np.concatenate([
            np.array(initialpos),
            np.array(initialor),
            np.array(initialvel),
            np.array(initialJointPoses),
            np.array(initialJointVelocities),
            np.array([self.angle]),
            np.array([initialisHolding])
        ])  # Total: 31 elements
        elif self.action_type=='pos_only':
            observation = np.concatenate([
            np.array(initialpos),
            np.array(initialor),
            np.array(initialvel),
            np.array(initialJointPoses),
            np.array(initialJointVelocities),
            np.array([self.pos_distance]),
            np.array([initialisHolding])
        ])
        else: 
            observation = np.concatenate([
                np.array(initialpos),
                np.array(initialor),
                np.array(initialvel),
                np.array(initialJointPoses),
                np.array(initialJointVelocities),
                np.array([self.pos_distance]),
                np.array([self.angle]),
                np.array([initialisHolding])
            ])
        self.isHolding = initialisHolding
        if self.action_type == 'ori_only':
            achieved_goal = np.array(list(initialor) +[self.isHolding])
            desired_goal = np.array(list(self.goal_ori) + [1])
        elif self.action_type == 'pos_only':
            achieved_goal = np.array(list(initialpos) + [self.isHolding])
            desired_goal = np.array(list(self.goal_pos) + [1])
        else:
            achieved_goal = np.array(list(initialpos) + list(initialor) + [self.isHolding])
            desired_goal = np.array(list(self.target_position) + [1])
        if self.obs_type == 'dict':
            observation_dict = {
                "observation": observation.astype(np.float32),
                "achieved_goal": achieved_goal.astype(np.float32),
                "desired_goal": desired_goal.astype(np.float32),
            }
            self.state = observation_dict
        else:
            self.state = observation.astype(np.float32)
        self.pos_action = []
        return self.state, {}

    

    def step(self, action):
        self.current_step += 1
        dx, dy, dz, qx, qy, qz, qw, x, y, z = unpack_action(self,action, self.dv)
        mode_map = {
            'rot_vec': 'rot_vec',
            'sixactions': 'sixactions',
            'fouractions': 'fouractions',
            'fiveactions': 'fiveactions',
            'quat': 'quat',
            'joint': 'joint',
            'ori_only': 'ori_only',
            'pos_only': 'pos_only'
        }
        mode = mode_map.get(self.action_type, None)

        if self.action_type == 'joint':
            jointPoses, _ = get_new_pose(self,dx, dy, dz, qx, qy, qz, None, mode)
        else:
            newPosition, newOrientation = get_new_pose(self,dx, dy, dz, qx, qy, qz, qw, mode)
            if self.action_type == 'pos_only':
                jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=newPosition, maxNumIterations=100, residualThreshold=1e-4)
            else:
                jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, targetPosition=newPosition, targetOrientation=newOrientation, maxNumIterations=100, residualThreshold=1e-4)
            if np.any(np.isnan(jointPoses)) or np.any(np.abs(jointPoses) > 10):
                print("IK failure, skipping step")
                # fallback or skip this step

        p.setJointMotorControlArray(self.pandaUid, list(range(9)), p.POSITION_CONTROL, list(jointPoses))
        for _ in range(20):
            p.stepSimulation()
            # time.sleep(1./500)  # Remove for speed

        actualNewPosition = p.getLinkState(self.pandaUid, 11)[0]
        actualNewOrientation = p.getLinkState(self.pandaUid, 11)[1]
        actualNewVelocity = p.getLinkState(self.pandaUid, 11, 1)[6]
        holdObject = len(p.getContactPoints(self.pandaUid, self.objectUid))
        self.isHolding = 1 if holdObject > 0 else 0
        joint_states = [p.getJointState(self.pandaUid, i) for i in range(9)]
        jointPoses = np.array([js[0] for js in joint_states])        # positions
        jointVelocities = np.array([js[1] for js in joint_states])   # velocities
        self.pos_distance, self.angle = calculate_distances(self, actualNewPosition, actualNewOrientation, self.goal_pos, self.goal_ori)
        
        if self.action_type == 'pos_only':
            observation_state = np.concatenate([
                np.array(actualNewPosition),
                np.array(actualNewOrientation),  # 4 elements
                np.array(actualNewVelocity),     # 3 elements
                np.array(jointPoses),            # 9 elements
                np.array(jointVelocities),       # 9 elements
                np.array([self.pos_distance]),   # 1 element
                np.array([self.isHolding])       # 1 element
            ])  # Total: 30 elements instead of 31
        elif self.action_type == 'ori_only':
            observation_state = np.concatenate([
                np.array(actualNewOrientation),
                np.array(actualNewPosition),  # 4 elements
                np.array(actualNewVelocity),     # 3 elements
                np.array(jointPoses),            # 9 elements
                np.array(jointVelocities),       # 9 elements
                np.array([self.angle]),          # 1 element
                np.array([self.isHolding])       # 1 element
            ])  # Total: 27 elements
        else:
            # Full observation for other action types
            observation_state = np.concatenate([
                np.array(actualNewPosition),
                np.array(actualNewOrientation),
                np.array(actualNewVelocity),
                np.array(jointPoses),
                np.array(jointVelocities),
                np.array([self.pos_distance]),
                np.array([self.angle]),
                np.array([self.isHolding])
            ])
        if self.action_type == 'ori_only':
            achieved_goal = np.array(list(actualNewOrientation) + [self.isHolding])
            desired_goal = np.array(list(self.goal_ori) + [1])
        elif self.action_type == 'pos_only':
            achieved_goal = np.array(list(actualNewPosition)+[self.isHolding])
            desired_goal = np.array(list(self.goal_pos) + [1])
        else:
            achieved_goal = np.array(list(actualNewPosition) + list(actualNewOrientation) + [self.isHolding])
            desired_goal = np.array(list(self.target_position) + [1])

        if self.obs_type == 'dict':
            observation = {
                "observation": observation_state.astype(np.float32),
                "achieved_goal": achieved_goal.astype(np.float32),
                "desired_goal": desired_goal.astype(np.float32),
            }
            self.state = observation
        else:
            self.state = observation_state.astype(np.float32)
        done = check_done(self)
        truncated = self.current_step >= self.max_steps and not done
        info = {'is_success': done, 'current_step': self.current_step}
        reward = self.compute_reward(achieved_goal, desired_goal, info)


        return self.state, np.float64(reward), done, truncated, info

    

    def close(self):
        if self.connected:
            p.disconnect()
            self.connected = False
