import numpy as np
from scipy.spatial.transform import Rotation as R
#
from high_mpc.common.quad_index import *

#
class Quadrotor_v0(object):
    #
    def __init__(self, dt):
        self.s_dim = 14
        self.a_dim = 4
        #
        self._state = np.zeros(shape=self.s_dim)
        self._state[kQuatW] = 1.0
        #
        self._actions = np.zeros(shape=self.a_dim)

        #
        self._gz = 9.81
        self._dt = dt
        self._arm_l = 0.3   # m
        
        # Sampling range of the quadrotor's initial position
        self._xyz_dist = np.array(
            [ [-1.0, -1.0], # x 
              [-1.0, 1.0],   # y
              [1.0, 1.0]]   # z
        )
        # Sampling range of the quadrotor's initial velocity
        self._vxyz_dist = np.array(
            [ [-1.0, 1.0],  # vx
              [-1.0, 1.0],  # vy
              [-1.0, 1.0]]  # vz
        )
        
        # x, y, z, r, p, y, vx, vy, vz
        self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10, -0.1, -0.1, -1.0, -1.0])
        self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10, 0.1, 0.1, 1.0, 1.0])
        #
        self.reset()
        # self._t = 0.0
        self.l = -0.1
        self.kosi = 0.1
    
    def reset(self):
        self._state = np.zeros(shape=self.s_dim)
        self._state[kQuatW] = 1.0 # 
        #
        # initialize position, randomly
        self._state[kPosX] = np.random.uniform(
            low=self._xyz_dist[0, 0], high=self._xyz_dist[0, 1])
        self._state[kPosY] = np.random.uniform(
            low=self._xyz_dist[1, 0], high=self._xyz_dist[1, 1])
        self._state[kPosZ] = np.random.uniform(
            low=self._xyz_dist[2, 0], high=self._xyz_dist[2, 1])
        
        # initialize rotation, randomly
        quad_quat0 = np.random.uniform(low=0.0, high=1, size=4)
        # normalize the quaternion
        self._state[kQuatW:kQuatZ+1] = quad_quat0 / np.linalg.norm(quad_quat0)
        
        # initialize velocity, randomly
        self._state[kVelX] = np.random.uniform(
            low=self._vxyz_dist[0, 0], high=self._vxyz_dist[0, 1])
        self._state[kVelY] = np.random.uniform(
            low=self._vxyz_dist[1, 0], high=self._vxyz_dist[1, 1])
        self._state[kVelZ] = np.random.uniform(
            low=self._vxyz_dist[2, 0], high=self._vxyz_dist[2, 1])
        #
        return self._state

    def run(self, action):
        """
        Apply the control command on the quadrotor and transits the system to the next state
        """
        # rk4 int
        M = 4
        DT = self._dt / M
        #
        X = self._state
        for i in range(M):
            k1 = DT*self._f(X, action)
            k2 = DT*self._f(X + 0.5*k1, action)
            k3 = DT*self._f(X + 0.5*k2, action)
            k4 = DT*self._f(X + k3, action)
            #
            X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0
        #
        self._state = X
        return self._state

    def _f(self, state, action):
        """
        System dynamics: ds = f(x, u)
        """
        thrust, wx, wy, wz = action
        #
        dstate = np.zeros(shape=self.s_dim)

        dstate[kPosX:kPosZ+1] = state[kVelX:kVelZ+1]

        qw, qx, qy, qz = self.get_quaternion()
        

        dstate[kQuatW] = 0.5 * ( -wx*qx - wy*qy - wz*qz )
        dstate[kQuatX] = 0.5 * (  wx*qw + wz*qy - wy*qz )
        dstate[kQuatY] = 0.5 * (  wy*qw - wz*qx + wx*qz )
        dstate[kQuatZ] = 0.5 * (  wz*qw + wy*qx - wx*qy )

        dstate[kVelX] = 2 * ( qw*qy + qx*qz ) * thrust
        dstate[kVelY] = 2 * ( qy*qz - qw*qx ) * thrust
        # dstate[kVelZ] = (1 - 2*qx*qx - 2*qy*qy) * thrust - self._gz
        dstate[kVelZ] = (qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz

        ax = dstate[kVelX]*0
        ay = dstate[kVelY]*0
        az = dstate[kVelZ]*0
        r = dstate[kr]
        s = dstate[ks]
        r_dot = dstate[kr_dot]
        s_dot = dstate[ks_dot]
        # self.kosi = (self.l**2-r**2-s**2)**0.5
        # dstate[kr_dot] = 0
        # dstate[ks_dot] = 0
        dstate[kr_dot] = (self.l**6*ax*self.kosi**2 - self.l**6*ay*r*s - 2*self.l**4*ax*self.kosi**2*r**2 - 3*self.l**4*ax*self.kosi**2*s**2 + 3*self.l**4*ay*r**3*s + 3*self.l**4*ay*r*s**3 - self.l**4*az*self.kosi**3*r + self.l**4*az*self.kosi*r*s**2 - self.l**4*self._gz*self.kosi**3*r + self.l**4*self._gz*self.kosi*r*s**2 + self.l**4*self.kosi**2*r*r_dot**2 + self.l**4*self.kosi**2*r*s_dot**2 - self.l**4*r*r_dot**2*s**2 - self.l**4*r*s**2*s_dot**2 + self.l**2*ax*self.kosi**2*r**4 + 4*self.l**2*ax*self.kosi**2*r**2*s**2 + 3*self.l**2*ax*self.kosi**2*s**4 - 3*self.l**2*ay*r**5*s - 6*self.l**2*ay*r**3*s**3 - 3*self.l**2*ay*r*s**5 + self.l**2*az*self.kosi**3*r**3 + 2*self.l**2*az*self.kosi**3*r*s**2 - 2*self.l**2*az*self.kosi*r**3*s**2 - 2*self.l**2*az*self.kosi*r*s**4 + self.l**2*self._gz*self.kosi**3*r**3 + 2*self.l**2*self._gz*self.kosi**3*r*s**2 - 2*self.l**2*self._gz*self.kosi*r**3*s**2 - 2*self.l**2*self._gz*self.kosi*r*s**4 - self.l**2*self.kosi**2*r**3*s_dot**2 + 2*self.l**2*self.kosi**2*r**2*r_dot*s*s_dot - 2*self.l**2*self.kosi**2*r*r_dot**2*s**2 - self.l**2*self.kosi**2*r*s**2*s_dot**2 + self.l**2*r**3*r_dot**2*s**2 + 2*self.l**2*r**3*s**2*s_dot**2 - 2*self.l**2*r**2*r_dot*s**3*s_dot + 2*self.l**2*r*r_dot**2*s**4 + self.l**2*r*s**4*s_dot**2 - ax*self.kosi**2*r**4*s**2 - 2*ax*self.kosi**2*r**2*s**4 - ax*self.kosi**2*s**6 + ay*r**7*s + 3*ay*r**5*s**3 + 3*ay*r**3*s**5 + ay*r*s**7 - az*self.kosi**3*r**3*s**2 - az*self.kosi**3*r*s**4 + az*self.kosi*r**5*s**2 + 2*az*self.kosi*r**3*s**4 + az*self.kosi*r*s**6 - self._gz*self.kosi**3*r**3*s**2 - self._gz*self.kosi**3*r*s**4 + self._gz*self.kosi*r**5*s**2 + 2*self._gz*self.kosi*r**3*s**4 + self._gz*self.kosi*r*s**6 + self.kosi**2*r**3*s**2*s_dot**2 - 2*self.kosi**2*r**2*r_dot*s**3*s_dot + self.kosi**2*r*r_dot**2*s**4 - r**5*s**2*s_dot**2 + 2*r**4*r_dot*s**3*s_dot - r**3*r_dot**2*s**4 - r**3*s**4*s_dot**2 + 2*r**2*r_dot*s**5*s_dot - r*r_dot**2*s**6)/(-self.l**4*self.kosi**4 + self.l**4*r*s**2 + 2*self.l**2*self.kosi**4*s**2 - self.l**2*r**4*s**2 - self.l**2*r**3*s**2 - self.l**2*r**2*s**4 - self.l**2*r*s**4 - self.kosi**4*s**4 + r**6*s**2 + 2*r**4*s**4 + r**2*s**6)
        dstate[ks_dot] = (-self.l**6*ax*s + self.l**6*ay*self.kosi**2 + self.l**4*ax*r**3*s + 2*self.l**4*ax*r**2*s + self.l**4*ax*r*s**3 + 2*self.l**4*ax*s**3 - 2*self.l**4*ay*self.kosi**2*r**2 - 3*self.l**4*ay*self.kosi**2*s**2 - self.l**4*az*self.kosi**3*s + self.l**4*az*self.kosi*r*s - self.l**4*self._gz*self.kosi**3*s + self.l**4*self._gz*self.kosi*r*s + self.l**4*self.kosi**2*r_dot**2*s + self.l**4*self.kosi**2*s*s_dot**2 - self.l**4*r*r_dot**2*s - self.l**4*r*s*s_dot**2 - 2*self.l**2*ax*r**5*s - self.l**2*ax*r**4*s - 4*self.l**2*ax*r**3*s**3 - 2*self.l**2*ax*r**2*s**3 - 2*self.l**2*ax*r*s**5 - self.l**2*ax*s**5 + self.l**2*ay*self.kosi**2*r**4 + 4*self.l**2*ay*self.kosi**2*r**2*s**2 + 3*self.l**2*ay*self.kosi**2*s**4 + self.l**2*az*self.kosi**3*r**2*s + 2*self.l**2*az*self.kosi**3*s**3 - self.l**2*az*self.kosi*r**4*s - self.l**2*az*self.kosi*r**3*s - self.l**2*az*self.kosi*r**2*s**3 - self.l**2*az*self.kosi*r*s**3 + self.l**2*self._gz*self.kosi**3*r**2*s + 2*self.l**2*self._gz*self.kosi**3*s**3 - self.l**2*self._gz*self.kosi*r**4*s - self.l**2*self._gz*self.kosi*r**3*s - self.l**2*self._gz*self.kosi*r**2*s**3 - self.l**2*self._gz*self.kosi*r*s**3 - self.l**2*self.kosi**2*r**2*s*s_dot**2 + 2*self.l**2*self.kosi**2*r*r_dot*s**2*s_dot - 2*self.l**2*self.kosi**2*r_dot**2*s**3 - self.l**2*self.kosi**2*s**3*s_dot**2 + self.l**2*r**4*r_dot**2*s + self.l**2*r**4*s*s_dot**2 + self.l**2*r**3*s*s_dot**2 + self.l**2*r**2*r_dot**2*s**3 - 2*self.l**2*r**2*r_dot*s**2*s_dot + self.l**2*r**2*s**3*s_dot**2 + self.l**2*r*r_dot**2*s**3 + ax*r**7*s + 3*ax*r**5*s**3 + 3*ax*r**3*s**5 + ax*r*s**7 - ay*self.kosi**2*r**4*s**2 - 2*ay*self.kosi**2*r**2*s**4 - ay*self.kosi**2*s**6 - az*self.kosi**3*r**2*s**3 - az*self.kosi**3*s**5 + az*self.kosi*r**6*s + 2*az*self.kosi*r**4*s**3 + az*self.kosi*r**2*s**5 - self._gz*self.kosi**3*r**2*s**3 - self._gz*self.kosi**3*s**5 + self._gz*self.kosi*r**6*s + 2*self._gz*self.kosi*r**4*s**3 + self._gz*self.kosi*r**2*s**5 + self.kosi**2*r**2*s**3*s_dot**2 - 2*self.kosi**2*r*r_dot*s**4*s_dot + self.kosi**2*r_dot**2*s**5 - r**6*s*s_dot**2 + 2*r**5*r_dot*s**2*s_dot - r**4*r_dot**2*s**3 - r**4*s**3*s_dot**2 + 2*r**3*r_dot*s**4*s_dot - r**2*r_dot**2*s**5)/(-self.l**4*self.kosi**4 + self.l**4*r*s**2 + 2*self.l**2*self.kosi**4*s**2 - self.l**2*r**4*s**2 - self.l**2*r**3*s**2 - self.l**2*r**2*s**4 - self.l**2*r*s**4 - self.kosi**4*s**4 + r**6*s**2 + 2*r**4*s**4 + r**2*s**6)
        

        return dstate

    def set_state(self, state):
        """
        Set the vehicle's state
        """
        self._state = state
        
    def get_state(self):
        """
        Get the vehicle's state
        """
        return self._state

    def get_cartesian_state(self):
        """
        Get the Full state in Cartesian coordinates
        """
        cartesian_state = np.zeros(shape=13)
        cartesian_state[0:3] = self.get_position()
        cartesian_state[3:6] = self.get_euler()
        cartesian_state[6:9] = self.get_velocity()
        # cartesian_state[10:14] = self.get_pendulum_below_state()
        cartesian_state[9:13] = self._state[kr:ks_dot+1]
        return cartesian_state
    
    def get_position(self,):
        """
        Retrieve Position
        """
        return self._state[kPosX:kPosZ+1]
    
    def get_velocity(self,):
        """
        Retrieve Linear Velocity
        """
        return self._state[kVelX:kVelZ+1]

    def get_pendulum_below_state(self,):
        """
        Retrieve pendulum Position(dx, dy), Velocity(vx, vy)
        """
        return self._state[kr:ks_dot+1]

    
    def get_quaternion(self,):
        """
        Retrieve Quaternion
        """
        quat = np.zeros(4)
        quat = self._state[kQuatW:kQuatZ+1]
        quat = quat / np.linalg.norm(quat)
        return quat

    def get_euler(self,):
        """
        Retrieve Euler Angles of the Vehicle
        """
        quat = self.get_quaternion()
        euler = self._quatToEuler(quat)
        return euler

    def get_axes(self):
        """
        Get the 3 axes (x, y, z) in world frame (for visualization only)
        """
        # axes in body frame
        b_x = np.array([self._arm_l, 0, 0])
        b_y = np.array([0, self._arm_l, 0])
        b_z = np.array([0, 0,  -self._arm_l])
        
        # rotation matrix
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()
        
        # axes in body frame
        w_x = rot_matrix@b_x + quad_center
        w_y = rot_matrix@b_y + quad_center
        w_z = rot_matrix@b_z + quad_center
        return [w_x, w_y, w_z]

    def get_motor_pos(self):
        """
        Get the 4 motor poses in world frame (for visualization only)
        """
        # motor position in body frame
        b_motor1 = np.array([np.sqrt(self._arm_l/2), np.sqrt(self._arm_l/2), 0])
        b_motor2 = np.array([-np.sqrt(self._arm_l/2), np.sqrt(self._arm_l/2), 0])
        b_motor3 = np.array([-np.sqrt(self._arm_l/2), -np.sqrt(self._arm_l/2), 0])
        b_motor4 = np.array([np.sqrt(self._arm_l/2), -np.sqrt(self._arm_l/2), 0])
        #
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()
        
        # motor position in world frame
        w_motor1 = rot_matrix@b_motor1 + quad_center
        w_motor2 = rot_matrix@b_motor2 + quad_center
        w_motor3 = rot_matrix@b_motor3 + quad_center
        w_motor4 = rot_matrix@b_motor4 + quad_center
        return [w_motor1, w_motor2, w_motor3, w_motor4]

    @staticmethod
    def _quatToEuler(quat):
        """
        Convert Quaternion to Euler Angles
        """
        quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
        euler_x = np.arctan2(2*quat_w*quat_x + 2*quat_y*quat_z, quat_w*quat_w - quat_x*quat_x - quat_y*quat_y + quat_z*quat_z)
        euler_y = -np.arcsin(2*quat_x*quat_z - 2*quat_w*quat_y)
        euler_z = np.arctan2(2*quat_w*quat_z+2*quat_x*quat_y, quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
        return [euler_x, euler_y, euler_z]
