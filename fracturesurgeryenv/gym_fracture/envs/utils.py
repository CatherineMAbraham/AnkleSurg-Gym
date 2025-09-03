import pybullet as p
import numpy as np
import time 
import pybullet_data
import os
from scipy.spatial.transform import Rotation as R
def make_scene(self):
    #Start Positions: Worked out previously
       startposition = np.array([0.03, 0.2, 0, -1.802, -2.89, 2.8, 0.61, 0.04, 0.04])

       #load scene
       #Make Plane, Table, Cube       
       plane_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,halfExtents=np.array([30.0, 30.0, 0.01]))
       plane_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=np.array([30.0, 30.0, 0.01]),rgbaColor=[0.678, 0.847, 0.902, 1])
       plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision_shape, 
                             baseVisualShapeIndex=plane_visual_shape,basePosition=[0, 0, -0.33])
       
       table =p.loadURDF("table/table.urdf", basePosition =[0.8,-0.32,-0.33], globalScaling =0.5);#[0.8, 0.4, -0.33]

       self.visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.01,0.01,0.01], rgbaColor=[0.835, 0.7216, 1, 1])  # Purple Goal box - no collision properties

       #Set up robot with calculated start positions
       urdfRootPath=pybullet_data.getDataPath()
      
       self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),
                                  basePosition=[0,-0.06,-0.33],#[-0.5,0,-0.65],
                                  useFixedBase=True, globalScaling = 1)
       
       p.changeDynamics(self.pandaUid,9, lateralFriction= 1,spinningFriction= 0.001)
       p.changeDynamics(self.pandaUid,10, lateralFriction= 1,spinningFriction= 0.001)
       p.resetJointState(self.pandaUid,9, 0.01)
       p.resetJointState(self.pandaUid,10, 0.01) 

       for i in range(8):
           p.resetJointState(self.pandaUid,i, startposition[i])
        
       for _ in range(10):
           p.stepSimulation()
           time.sleep(0.002)

           

       return self.pandaUid

def getGoal(self, fracturestart, fractureorientaionDeg):
    self.goal_range_low = fracturestart- [0.0125,0.01,0.003]
    self.goal_range_high = fracturestart+[0.0125,0.02,0.003]
    self.goal_ori_low= np.radians(fractureorientaionDeg - [15,5,15])
    self.goal_ori_high=np.radians(fractureorientaionDeg + [15,5,15])
    goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high,))
    
    if self.action_type == 'fiveactions' or self.action_type== 'fouractions':
        goal_pos[2] = fracturestart[2]
   
    if self.action_type == 'fouractions':
        self.goal_ori_low[1] =np.radians(fractureorientaionDeg[1] - 0)
        self.goal_ori_high[1] =np.radians(fractureorientaionDeg[1]+0)    
    
    self.goal_pos = np.round(goal_pos,3)
    ori = np.array(self.np_random.uniform(self.goal_ori_low, self.goal_ori_high))
    goal_ori = np.array(p.getQuaternionFromEuler(ori))
    #goal_ori = R.from_euler('xyz', ori).as_quat()
    self.goal_ori = np.round(goal_ori,3)


def getStarts(self):
    fracturestart= np.array(p.getLinkState(self.pandaUid, 11)[0] )
    fractureorientaionRad =p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])
    fractureorientaionDeg = np.degrees(np.array(fractureorientaionRad)) 
    
    #Calculated this difference from the object start position
    difference = [-0.004493, 0.079895+0.005, 0.073322]
    difference =np.array(difference)
    legstart=[]
    for i in range(len(difference)):
        leg = (fracturestart[i])-(difference[i])
        legstart.append(leg)
        
        i+=1
    

    return fracturestart, fractureorientaionDeg, legstart

def check_done(self):
        if self.horizon == 'variable' and self.action_type not in ['ori_only', 'pos_only']:
            return self.pos_distance <= self.distance_threshold_pos and self.angle <= self.distance_threshold_ori and self.isHolding == 1
        elif self.horizon == 'fixed' and self.action_type == 'ori_only':
            return self.angle <= self.distance_threshold_ori and self.isHolding == 1 and self.current_step >= self.max_steps
        elif self.horizon == 'fixed' and self.action_type == 'pos_only':
            return self.pos_distance <= self.distance_threshold_pos and self.isHolding == 1 and self.current_step >= self.max_steps
        elif self.action_type == 'ori_only':
            return self.angle <= self.distance_threshold_ori and self.isHolding == 1
        elif self.action_type == 'pos_only':
            return self.pos_distance <= self.distance_threshold_pos and self.isHolding == 1
        else:
            return self.pos_distance <= self.distance_threshold_pos and self.angle <= self.distance_threshold_ori and self.isHolding == 1

def get_new_pose(self, dx, dy, dz, qx, qy, qz, qw=None, mode=None):
        currentPose = p.getLinkState(self.pandaUid, 11, 1)
        currentPosition = np.array(currentPose[0])
        currentOrientation = np.array(currentPose[1])

        if mode == 'rot_vec':
            rotation_vector = np.array([qx, qy, qz])
            angle = np.linalg.norm(rotation_vector)
            if angle < 1e-6:
                deltaOr = [0, 0, 0, 1]
            else:
                max_rotation = np.deg2rad(1)
                clipped_angle = min(angle, max_rotation)
                axis = rotation_vector / angle
                deltaOr = p.getQuaternionFromAxisAngle(axis, clipped_angle)
            deltaPos = [dx, dy, dz]
            newPosition, newOrientation = p.multiplyTransforms(currentPosition, currentOrientation, deltaPos, deltaOr)
            newPosition = np.clip(newPosition, self.goal_range_low, self.goal_range_high)
            return newPosition, newOrientation

        elif mode in ['sixactions', 'fouractions', 'fiveactions', 'ori_only']:
            deltaorE = [qx, qy, qz]
            deltaor = p.getQuaternionFromEuler(deltaorE)
            if mode == 'ori_only':
                newPosition = currentPosition
            else:
                newPosition = currentPosition + np.array([dx, dy, dz])
            newPosition = np.clip(newPosition, self.goal_range_low, self.goal_range_high)
            newOrientation = np.array(p.multiplyTransforms([0, 0, 0], currentOrientation, [0, 0, 0], deltaor)[1])
            euler = p.getEulerFromQuaternion(newOrientation)
            newOrientationE = np.clip(euler, self.goal_ori_low, self.goal_ori_high)
            newOrientation = p.getQuaternionFromEuler(newOrientationE)
            return newPosition, newOrientation

        elif mode == 'quat':
            deltaOr = np.array([qx, qy, qz, qw])
            deltaOr /= np.linalg.norm(deltaOr)
            axis, angle = p.getAxisAngleFromQuaternion(deltaOr)
            max_rotation = np.deg2rad(3)
            if angle > 0:
                clipped_angle = min(angle, max_rotation)
                deltaOr = p.getQuaternionFromAxisAngle(axis, clipped_angle)
            else:
                deltaOr = [0, 0, 0, 1]
            deltaPos = [dx, dy, dz]
            newPosition, newOrientation = p.multiplyTransforms(currentPosition, currentOrientation, deltaPos, deltaOr)
            newPosition = np.clip(newPosition, self.goal_range_low, self.goal_range_high)
            return newPosition, newOrientation

        elif mode == 'pos_only':
            newPosition = currentPosition + np.array([qx, qy, qz])
            newOrientation = currentOrientation
            #newPosition[2] = np.clip(newPosition[2], self.goal_range_low[2], self.goal_range_high[2])
            newPosition = np.clip(newPosition, (self.goal_range_low), (self.goal_range_high))
            return newPosition, newOrientation

        elif mode == 'joint':
            currentJointPoses = [p.getJointState(self.pandaUid, i)[0] for i in range(9)]
            jointPoses = np.array(currentJointPoses) + np.array([dx, dy, dz, qx, qy, qz, 0, 0, 0])
            return jointPoses, None

        else:
            newPosition = currentPosition + np.array([dx, dy, dz])
            newPosition = np.clip(newPosition, self.goal_range_low, self.goal_range_high)
            newOrientation = np.array([qx, qy, qz])
            return newPosition, newOrientation

def unpack_action(self, action, dv):
    zeros = [0] * 10
    if self.action_type in ['ori_only', 'pos_only']:
        return [0, 0, 0, action[0] * dv, action[1] * dv, action[2] * dv, 0, 0, 0, 0]
    elif self.action_type == 'quat':
        return [action[0] * dv, action[1] * dv, action[2] * dv, action[3] * dv, action[4] * dv, action[5] * dv, action[6] * dv, 0, 0, 0]
    elif self.action_type == 'joint':
        return [action[0] * dv, action[1] * dv, action[2] * dv, action[3] * dv, action[4] * dv, action[5] * dv, action[6] * dv, action[6] * dv, action[7] * dv, action[8] * dv]
    elif self.action_type == 'fiveactions':
        return [action[0] * dv, action[1] * dv, 0, action[2] * dv, action[3] * dv, action[4] * dv, 0, 0, 0, 0]
    elif self.action_type == 'fouractions':
        return [action[0] * dv, action[1] * dv, 0, action[2] * dv, 0, action[3] * dv, 0, 0, 0, 0]
    else:
        return [action[0] * dv, action[1] * dv, action[2] * dv, action[3] * dv, action[4] * dv, action[5] * dv, 0, 0, 0, 0]


def calculate_distances(self,new_pos,new_ori,goal_pos,goal_ori):
    # Calculate positional distance (Euclidean distance)
    self.pos_distance = (np.linalg.norm(np.array(new_pos) - np.array(goal_pos), axis=-1)) #the new distance
    
    # Calculate the dot product between the quaternions
    dot_product = np.abs(np.sum(new_ori * goal_ori, axis=-1))
    
    
    # Ensure the dot product is within the valid range for acos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle (rotational distance) between the quaternions
    self.angle = 2 * np.arccos(dot_product)
    
    return self.pos_distance, self.angle


