"""
Standard MPC for Passing through a dynamic gate
"""
import casadi as ca
import numpy as np
import time
from os import system
#
from high_mpc.common.quad_index import *

#
class MPC(object):
    """
    Nonlinear MPC
    """
    def __init__(self, T, dt, so_path='./nmpc.so'):
        """
        Nonlinear MPC for quadrotor control        
        """
        self.so_path = so_path

        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T/self._dt)

        # Gravity
        self._gz = 9.81
        self.l = -0.1
        self.kosi = 0.1

        # Quadrotor constant
        self._w_max_yaw = 6.0
        self._w_max_xy = 6.0
        self._thrust_min = 2.0
        self._thrust_max = 20.0 #20.0

        #
        # state dimension (px, py, pz,           # quadrotor position
        #                  qw, qx, qy, qz,       # quadrotor quaternion
        #                  vx, vy, vz,           # quadrotor linear velocity
        self._s_dim = 14
        # action dimensions (c_thrust, wx, wy, wz)
        self._u_dim = 4
        
        # cost matrix for tracking the goal point
        self._Q_goal = np.diag([
            100, 100, 100,  # delta_x, delta_y, delta_z
            10, 10, 10, 10, # delta_qw, delta_qx, delta_qy, delta_qz
            10, 10, 10, 
            0, 0, 0, 0]) 

        # cost matrix for tracking the pendulum motion
        self._Q_pen = np.diag([
            0, 100, 100,  # delta_x, delta_y, delta_z
            10, 10, 10, 10, # delta_qw, delta_qx, delta_qy, delta_qz
            0, 10, 10, # delta_vx, delta_vy, delta_vz
            0, 0, 0, 0]) 
        
        # cost matrix for the action
        self._Q_u = np.diag([0.1, 0.1, 0.1, 0.1]) # T, wx, wy, wz

        # initial state and control action
        self._quad_s0 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._quad_u0 = [9.81, 0.0, 0.0, 0.0]

        self._initDynamics()

    def _initDynamics(self,):
        # # # # # # # # # # # # # # # # # # # 
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # # 

        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        #
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), \
            ca.SX.sym('qz')
        #
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')

        r, s = ca.SX.sym('r'), ca.SX.sym('s') #########
        r_dot, s_dot = ca.SX.sym('r_dot'), ca.SX.sym('s_dot')

        # -- conctenated vector
        self._x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz, r, s, r_dot, s_dot) 


        # # # # # # # # # # # # # # # # # # # 
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        thrust, wx, wy, wz = ca.SX.sym('thrust'), ca.SX.sym('wx'), \
            ca.SX.sym('wy'), ca.SX.sym('wz')
        
        # -- conctenated vector
        self._u = ca.vertcat(thrust, wx, wy, wz)
        
        # # # # # # # # # # # # # # # # # # # 
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #
        # ax = (2 * ( qw*qy + qx*qz ) * thrust )
        # ay = (2 * ( qy*qz - qw*qx ) * thrust)
        # az= ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)
        # self.kosi = (self.l**2-r**2-s**2)**0.5

        # den = 1/(self.l**2-s**2)/self.kosi**2
        
        # num1 = -r**4*ax - (self.l**2-s**2)**2*ax - 2*r**2*(s*r_dot*s_dot+(-self.l**2+s**2)*ax) + \
        # r**3*(s_dot**2+s*s_dot2-self.kosi*(g+az)) + r*(-self.l**2*s*s_dot2 + s**3*s_dot2 + s**2*(r_dot**2-self.kosi*(g+az)) + \
        # self.l**2*(-r_dot**2-s_dot**2+self.kosi*(g+az)))

        # num2 = -s**4*ay - (self.l**2-r**2)**2*ay - 2*s**2*(r*r_dot*s_dot+(-self.l**2+r**2)*ay) + \
        # s**3*(r_dot**2+r*r_dot2-self.kosi*(g+az)) + s*(-self.l**2*r_dot2+r**3*r_dot2+r**2*(s_dot**2-self.kosi*(g+az)) + \
        # self.l**2*(-r_dot**2-s_dot**2+self.kosi*(g+az))) 

        x_dot = ca.vertcat(
            vx,
            vy,
            vz,
            0.5 * ( -wx*qx - wy*qy - wz*qz ),
            0.5 * (  wx*qw + wz*qy - wy*qz ),
            0.5 * (  wy*qw - wz*qx + wx*qz ),
            0.5 * (  wz*qw + wy*qx - wx*qy ),
            2 * ( qw*qy + qx*qz ) * thrust, # ax
            2 * ( qy*qz - qw*qx ) * thrust, # ay
            (qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz, # az
            # (1 - 2*qx*qx - 2*qy*qy) * thrust - self._gz
            r_dot,
            s_dot,
            # 0,
            # 0,
            # 0,
            (self.l**6*0*self.kosi**2 - self.l**6*0*r*s - 2*self.l**4*0*self.kosi**2*r**2 - 3*self.l**4*0*self.kosi**2*s**2 + 3*self.l**4*0*r**3*s + 3*self.l**4*0*r*s**3 - self.l**4*0*self.kosi**3*r + self.l**4*0*self.kosi*r*s**2 - self.l**4*self._gz*self.kosi**3*r + self.l**4*self._gz*self.kosi*r*s**2 + self.l**4*self.kosi**2*r*r_dot**2 + self.l**4*self.kosi**2*r*s_dot**2 - self.l**4*r*r_dot**2*s**2 - self.l**4*r*s**2*s_dot**2 + self.l**2*0*self.kosi**2*r**4 + 4*self.l**2*0*self.kosi**2*r**2*s**2 + 3*self.l**2*0*self.kosi**2*s**4 - 3*self.l**2*0*r**5*s - 6*self.l**2*0*r**3*s**3 - 3*self.l**2*0*r*s**5 + self.l**2*0*self.kosi**3*r**3 + 2*self.l**2*0*self.kosi**3*r*s**2 - 2*self.l**2*0*self.kosi*r**3*s**2 - 2*self.l**2*0*self.kosi*r*s**4 + self.l**2*self._gz*self.kosi**3*r**3 + 2*self.l**2*self._gz*self.kosi**3*r*s**2 - 2*self.l**2*self._gz*self.kosi*r**3*s**2 - 2*self.l**2*self._gz*self.kosi*r*s**4 - self.l**2*self.kosi**2*r**3*s_dot**2 + 2*self.l**2*self.kosi**2*r**2*r_dot*s*s_dot - 2*self.l**2*self.kosi**2*r*r_dot**2*s**2 - self.l**2*self.kosi**2*r*s**2*s_dot**2 + self.l**2*r**3*r_dot**2*s**2 + 2*self.l**2*r**3*s**2*s_dot**2 - 2*self.l**2*r**2*r_dot*s**3*s_dot + 2*self.l**2*r*r_dot**2*s**4 + self.l**2*r*s**4*s_dot**2 - 0*self.kosi**2*r**4*s**2 - 2*0*self.kosi**2*r**2*s**4 - 0*self.kosi**2*s**6 + 0*r**7*s + 3*0*r**5*s**3 + 3*0*r**3*s**5 + 0*r*s**7 - 0*self.kosi**3*r**3*s**2 - 0*self.kosi**3*r*s**4 + 0*self.kosi*r**5*s**2 + 2*0*self.kosi*r**3*s**4 + 0*self.kosi*r*s**6 - self._gz*self.kosi**3*r**3*s**2 - self._gz*self.kosi**3*r*s**4 + self._gz*self.kosi*r**5*s**2 + 2*self._gz*self.kosi*r**3*s**4 + self._gz*self.kosi*r*s**6 + self.kosi**2*r**3*s**2*s_dot**2 - 2*self.kosi**2*r**2*r_dot*s**3*s_dot + self.kosi**2*r*r_dot**2*s**4 - r**5*s**2*s_dot**2 + 2*r**4*r_dot*s**3*s_dot - r**3*r_dot**2*s**4 - r**3*s**4*s_dot**2 + 2*r**2*r_dot*s**5*s_dot - r*r_dot**2*s**6)/(-self.l**4*self.kosi**4 + self.l**4*r*s**2 + 2*self.l**2*self.kosi**4*s**2 - self.l**2*r**4*s**2 - self.l**2*r**3*s**2 - self.l**2*r**2*s**4 - self.l**2*r*s**4 - self.kosi**4*s**4 + r**6*s**2 + 2*r**4*s**4 + r**2*s**6),
            (-self.l**6*0*s + self.l**6*0*self.kosi**2 + self.l**4*0*r**3*s + 2*self.l**4*0*r**2*s + self.l**4*0*r*s**3 + 2*self.l**4*0*s**3 - 2*self.l**4*0*self.kosi**2*r**2 - 3*self.l**4*0*self.kosi**2*s**2 - self.l**4*0*self.kosi**3*s + self.l**4*0*self.kosi*r*s - self.l**4*self._gz*self.kosi**3*s + self.l**4*self._gz*self.kosi*r*s + self.l**4*self.kosi**2*r_dot**2*s + self.l**4*self.kosi**2*s*s_dot**2 - self.l**4*r*r_dot**2*s - self.l**4*r*s*s_dot**2 - 2*self.l**2*0*r**5*s - self.l**2*0*r**4*s - 4*self.l**2*0*r**3*s**3 - 2*self.l**2*0*r**2*s**3 - 2*self.l**2*0*r*s**5 - self.l**2*0*s**5 + self.l**2*0*self.kosi**2*r**4 + 4*self.l**2*0*self.kosi**2*r**2*s**2 + 3*self.l**2*0*self.kosi**2*s**4 + self.l**2*0*self.kosi**3*r**2*s + 2*self.l**2*0*self.kosi**3*s**3 - self.l**2*0*self.kosi*r**4*s - self.l**2*0*self.kosi*r**3*s - self.l**2*0*self.kosi*r**2*s**3 - self.l**2*0*self.kosi*r*s**3 + self.l**2*self._gz*self.kosi**3*r**2*s + 2*self.l**2*self._gz*self.kosi**3*s**3 - self.l**2*self._gz*self.kosi*r**4*s - self.l**2*self._gz*self.kosi*r**3*s - self.l**2*self._gz*self.kosi*r**2*s**3 - self.l**2*self._gz*self.kosi*r*s**3 - self.l**2*self.kosi**2*r**2*s*s_dot**2 + 2*self.l**2*self.kosi**2*r*r_dot*s**2*s_dot - 2*self.l**2*self.kosi**2*r_dot**2*s**3 - self.l**2*self.kosi**2*s**3*s_dot**2 + self.l**2*r**4*r_dot**2*s + self.l**2*r**4*s*s_dot**2 + self.l**2*r**3*s*s_dot**2 + self.l**2*r**2*r_dot**2*s**3 - 2*self.l**2*r**2*r_dot*s**2*s_dot + self.l**2*r**2*s**3*s_dot**2 + self.l**2*r*r_dot**2*s**3 + 0*r**7*s + 3*0*r**5*s**3 + 3*0*r**3*s**5 + 0*r*s**7 - 0*self.kosi**2*r**4*s**2 - 2*0*self.kosi**2*r**2*s**4 - 0*self.kosi**2*s**6 - 0*self.kosi**3*r**2*s**3 - 0*self.kosi**3*s**5 + 0*self.kosi*r**6*s + 2*0*self.kosi*r**4*s**3 + 0*self.kosi*r**2*s**5 - self._gz*self.kosi**3*r**2*s**3 - self._gz*self.kosi**3*s**5 + self._gz*self.kosi*r**6*s + 2*self._gz*self.kosi*r**4*s**3 + self._gz*self.kosi*r**2*s**5 + self.kosi**2*r**2*s**3*s_dot**2 - 2*self.kosi**2*r*r_dot*s**4*s_dot + self.kosi**2*r_dot**2*s**5 - r**6*s*s_dot**2 + 2*r**5*r_dot*s**2*s_dot - r**4*r_dot**2*s**3 - r**4*s**3*s_dot**2 + 2*r**3*r_dot*s**4*s_dot - r**2*r_dot**2*s**5)/(-self.l**4*self.kosi**4 + self.l**4*r*s**2 + 2*self.l**2*self.kosi**4*s**2 - self.l**2*r**4*s**2 - self.l**2*r**3*s**2 - self.l**2*r**2*s**4 - self.l**2*r*s**4 - self.kosi**4*s**4 + r**6*s**2 + 2*r**4*s**4 + r**2*s**6)
            # (self.l**6*(2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2 - self.l**6*(2 * ( qy*qz - qw*qx ) * thrust)*r*s - 2*self.l**4*(2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2*r**2 - 3*self.l**4*(2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2*s**2 + 3*self.l**4*(2 * ( qy*qz - qw*qx ) * thrust)*r**3*s + 3*self.l**4*(2 * ( qy*qz - qw*qx ) * thrust)*r*s**3 - self.l**4*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*r + self.l**4*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r*s**2 - self.l**4*self._gz*self.kosi**3*r + self.l**4*self._gz*self.kosi*r*s**2 + self.l**4*self.kosi**2*r*r_dot**2 + self.l**4*self.kosi**2*r*s_dot**2 - self.l**4*r*r_dot**2*s**2 - self.l**4*r*s**2*s_dot**2 + self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2*r**4 + 4*self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2*r**2*s**2 + 3*self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2*s**4 - 3*self.l**2*(2 * ( qy*qz - qw*qx ) * thrust)*r**5*s - 6*self.l**2*(2 * ( qy*qz - qw*qx ) * thrust)*r**3*s**3 - 3*self.l**2*(2 * ( qy*qz - qw*qx ) * thrust)*r*s**5 + self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*r**3 + 2*self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*r*s**2 - 2*self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**3*s**2 - 2*self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r*s**4 + self.l**2*self._gz*self.kosi**3*r**3 + 2*self.l**2*self._gz*self.kosi**3*r*s**2 - 2*self.l**2*self._gz*self.kosi*r**3*s**2 - 2*self.l**2*self._gz*self.kosi*r*s**4 - self.l**2*self.kosi**2*r**3*s_dot**2 + 2*self.l**2*self.kosi**2*r**2*r_dot*s*s_dot - 2*self.l**2*self.kosi**2*r*r_dot**2*s**2 - self.l**2*self.kosi**2*r*s**2*s_dot**2 + self.l**2*r**3*r_dot**2*s**2 + 2*self.l**2*r**3*s**2*s_dot**2 - 2*self.l**2*r**2*r_dot*s**3*s_dot + 2*self.l**2*r*r_dot**2*s**4 + self.l**2*r*s**4*s_dot**2 - (2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2*r**4*s**2 - 2*(2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2*r**2*s**4 - (2 * ( qw*qy + qx*qz ) * thrust )*self.kosi**2*s**6 + (2 * ( qy*qz - qw*qx ) * thrust)*r**7*s + 3*(2 * ( qy*qz - qw*qx ) * thrust)*r**5*s**3 + 3*(2 * ( qy*qz - qw*qx ) * thrust)*r**3*s**5 + (2 * ( qy*qz - qw*qx ) * thrust)*r*s**7 - ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*r**3*s**2 - ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*r*s**4 + ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**5*s**2 + 2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**3*s**4 + ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r*s**6 - self._gz*self.kosi**3*r**3*s**2 - self._gz*self.kosi**3*r*s**4 + self._gz*self.kosi*r**5*s**2 + 2*self._gz*self.kosi*r**3*s**4 + self._gz*self.kosi*r*s**6 + self.kosi**2*r**3*s**2*s_dot**2 - 2*self.kosi**2*r**2*r_dot*s**3*s_dot + self.kosi**2*r*r_dot**2*s**4 - r**5*s**2*s_dot**2 + 2*r**4*r_dot*s**3*s_dot - r**3*r_dot**2*s**4 - r**3*s**4*s_dot**2 + 2*r**2*r_dot*s**5*s_dot - r*r_dot**2*s**6)/(-self.l**4*self.kosi**4 + self.l**4*r*s**2 + 2*self.l**2*self.kosi**4*s**2 - self.l**2*r**4*s**2 - self.l**2*r**3*s**2 - self.l**2*r**2*s**4 - self.l**2*r*s**4 - self.kosi**4*s**4 + r**6*s**2 + 2*r**4*s**4 + r**2*s**6),
            # (-self.l**6*(2 * ( qw*qy + qx*qz ) * thrust )*s + self.l**6*(2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2 + self.l**4*(2 * ( qw*qy + qx*qz ) * thrust )*r**3*s + 2*self.l**4*(2 * ( qw*qy + qx*qz ) * thrust )*r**2*s + self.l**4*(2 * ( qw*qy + qx*qz ) * thrust )*r*s**3 + 2*self.l**4*(2 * ( qw*qy + qx*qz ) * thrust )*s**3 - 2*self.l**4*(2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2*r**2 - 3*self.l**4*(2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2*s**2 - self.l**4*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*s + self.l**4*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r*s - self.l**4*self._gz*self.kosi**3*s + self.l**4*self._gz*self.kosi*r*s + self.l**4*self.kosi**2*r_dot**2*s + self.l**4*self.kosi**2*s*s_dot**2 - self.l**4*r*r_dot**2*s - self.l**4*r*s*s_dot**2 - 2*self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*r**5*s - self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*r**4*s - 4*self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*r**3*s**3 - 2*self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*r**2*s**3 - 2*self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*r*s**5 - self.l**2*(2 * ( qw*qy + qx*qz ) * thrust )*s**5 + self.l**2*(2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2*r**4 + 4*self.l**2*(2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2*r**2*s**2 + 3*self.l**2*(2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2*s**4 + self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*r**2*s + 2*self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*s**3 - self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**4*s - self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**3*s - self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**2*s**3 - self.l**2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r*s**3 + self.l**2*self._gz*self.kosi**3*r**2*s + 2*self.l**2*self._gz*self.kosi**3*s**3 - self.l**2*self._gz*self.kosi*r**4*s - self.l**2*self._gz*self.kosi*r**3*s - self.l**2*self._gz*self.kosi*r**2*s**3 - self.l**2*self._gz*self.kosi*r*s**3 - self.l**2*self.kosi**2*r**2*s*s_dot**2 + 2*self.l**2*self.kosi**2*r*r_dot*s**2*s_dot - 2*self.l**2*self.kosi**2*r_dot**2*s**3 - self.l**2*self.kosi**2*s**3*s_dot**2 + self.l**2*r**4*r_dot**2*s + self.l**2*r**4*s*s_dot**2 + self.l**2*r**3*s*s_dot**2 + self.l**2*r**2*r_dot**2*s**3 - 2*self.l**2*r**2*r_dot*s**2*s_dot + self.l**2*r**2*s**3*s_dot**2 + self.l**2*r*r_dot**2*s**3 + (2 * ( qw*qy + qx*qz ) * thrust )*r**7*s + 3*(2 * ( qw*qy + qx*qz ) * thrust )*r**5*s**3 + 3*(2 * ( qw*qy + qx*qz ) * thrust )*r**3*s**5 + (2 * ( qw*qy + qx*qz ) * thrust )*r*s**7 - (2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2*r**4*s**2 - 2*(2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2*r**2*s**4 - (2 * ( qy*qz - qw*qx ) * thrust)*self.kosi**2*s**6 - ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*r**2*s**3 - ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi**3*s**5 + ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**6*s + 2*((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**4*s**3 + ((qw*qw - qx*qx -qy*qy + qz*qz) * thrust - self._gz)*self.kosi*r**2*s**5 - self._gz*self.kosi**3*r**2*s**3 - self._gz*self.kosi**3*s**5 + self._gz*self.kosi*r**6*s + 2*self._gz*self.kosi*r**4*s**3 + self._gz*self.kosi*r**2*s**5 + self.kosi**2*r**2*s**3*s_dot**2 - 2*self.kosi**2*r*r_dot*s**4*s_dot + self.kosi**2*r_dot**2*s**5 - r**6*s*s_dot**2 + 2*r**5*r_dot*s**2*s_dot - r**4*r_dot**2*s**3 - r**4*s**3*s_dot**2 + 2*r**3*r_dot*s**4*s_dot - r**2*r_dot**2*s**5)/(-self.l**4*self.kosi**4 + self.l**4*r*s**2 + 2*self.l**2*self.kosi**4*s**2 - self.l**2*r**4*s**2 - self.l**2*r**3*s**2 - self.l**2*r**2*s**4 - self.l**2*r*s**4 - self.kosi**4*s**4 + r**6*s**2 + 2*r**4*s**4 + r**2*s**6)
       
        )

        #
        self.f = ca.Function('f', [self._x, self._u], [x_dot], ['x', 'u'], ['ode'])
                
        # # Fold # discretization of dynamic model
        # fmap is used to get next state
        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, "openmp") # parallel
        
        # ca.subject_to(r**2+s**2<=self.l**2)
        # # # # # # # # # # # # # # # 
        # ---- loss function --------
        # # # # # # # # # # # # # # # 

        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self._s_dim)
        Delta_p = ca.SX.sym("Delta_p", self._s_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)        
        
        #        
        cost_goal = Delta_s.T @ self._Q_goal @ Delta_s 
        cost_gap = Delta_p.T @ self._Q_pen @ Delta_p 
        cost_u = Delta_u.T @ self._Q_u @ Delta_u

        #
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_goal])
        f_cost_gap = ca.Function('cost_gap', [Delta_p], [cost_gap])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        #
        # # # # # # # # # # # # # # # # # # # # 
        # # ---- Non-linear Optimization -----
        # # # # # # # # # # # # # # # # # # # #
        self.nlp_w = []       # nlp variables
        self.nlp_w0 = []      # initial guess of nlp variables
        self.lbw = []         # lower bound of the variables, lbw <= nlp_x
        self.ubw = []         # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0      # objective 
        self.nlp_g = []       # constraint functions
        self.lbg = []         # lower bound of constrait functions, lbg < g
        self.ubg = []         # upper bound of constrait functions, g < ubg

        u_min = [self._thrust_min, -self._w_max_xy, -self._w_max_xy, -self._w_max_yaw]
        u_max = [self._thrust_max,  self._w_max_xy,  self._w_max_xy,  self._w_max_yaw]
        x_bound = ca.inf
        # x_min = [-x_bound for _ in range(self._s_dim)]
        # x_max = [+x_bound for _ in range(self._s_dim)]
        x_min = [-x_bound for _ in range(10)] + [-(0.5*self.l**2)**0.5]*2 + [-x_bound for _ in range(2)]
        x_max = [+x_bound for _ in range(10)] + [(0.5*self.l**2)**0.5]*2 + [x_bound for _ in range(2)]
        #
        # g_min = [0 for _ in range(self._s_dim)]
        # g_max = [0 for _ in range(self._s_dim)]
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        P = ca.SX.sym("P", self._s_dim+(self._s_dim+3)*self._N+self._s_dim)
        X = ca.SX.sym("X", self._s_dim, self._N+1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        #
        X_next = fMap(X[:, :self._N], U)
        
        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._quad_s0
        self.lbw += x_min
        self.ubw += x_max
        
        # # starting point.
        self.nlp_g += [ X[:, 0] - P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max
        
        for k in range(self._N):
            #
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._quad_u0
            self.lbw += u_min
            self.ubw += u_max
            
            # retrieve time constant
            # idx_k = self._s_dim+self._s_dim+(self._s_dim+3)*(k)
            # idx_k_end = self._s_dim+(self._s_dim+3)*(k+1)
            # time_k = P[ idx_k : idx_k_end]

            # cost for tracking the goal position
            cost_goal_k, cost_gap_k = 0, 0
            if k >= self._N-1: # The goal postion.
                delta_s_k = (X[:, k+1] - P[self._s_dim+(self._s_dim+3)*self._N:])
                cost_goal_k = f_cost_goal(delta_s_k)
            else:
                # cost for tracking the moving gap
                delta_p_k = (X[:, k+1] - P[self._s_dim+(self._s_dim+3)*k : \
                    self._s_dim+(self._s_dim+3)*(k+1)-3]) 
                cost_gap_k = f_cost_gap(delta_p_k)
        
            delta_u_k = U[:, k]-[self._gz, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k +  cost_gap_k 

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k+1]]
            self.nlp_w0 += self._quad_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k+1]]
            self.lbg += g_min
            self.ubg += g_max
        
        print('mpc_obj = ', self.mpc_obj.shape)
        print('len(P) = ', P.shape)
        # nlp objective
        nlp_dict = {'f': self.mpc_obj, 
            'x': ca.vertcat(*self.nlp_w), 
            'p': P,               
            'g': ca.vertcat(*self.nlp_g) }        
        
        # # # # # # # # # # # # # # # # # # # 
        # -- qpoases            
        # # # # # # # # # # # # # # # # # # # 
        # nlp_options ={
        #     'verbose': False, \
        #     "qpsol": "qpoases", \
        #     "hessian_approximation": "gauss-newton", \
        #     "max_iter": 100, 
        #     "tol_du": 1e-2,
        #     "tol_pr": 1e-2,
        #     "qpsol_options": {"sparse":True, "hessian_type": "posdef", "numRefinementSteps":1} 
        # }
        # self.solver = ca.nlpsol("solver", "sqpmethod", nlp_dict, nlp_options)
        # cname = self.solver.generate_dependencies("mpc_v1.c")  
        # system('gcc -fPIC -shared ' + cname + ' -o ' + self.so_path)
        # self.solver = ca.nlpsol("solver", "sqpmethod", self.so_path, nlp_options)
        

        # # # # # # # # # # # # # # # # # # # 
        # -- ipopt
        # # # # # # # # # # # # # # # # # # # 
        ipopt_options = {
            'verbose': False, \
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0, 
            "print_time": False
        }
        
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)
        # # jit (just-in-time compilation)
        # print("Generating shared library........")
        # cname = self.solver.generate_dependencies("mpc_v1.c")  
        # system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + self.so_path) # -O3
        
        # # reload compiled mpc
        # print(self.so_path)
        
        # self.solver = ca.nlpsol("solver", "ipopt", self.so_path, ipopt_options)

    def solve(self, ref_states):
        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
        print('ref_states length = ',len(ref_states))
        self.sol = self.solver(
            x0=self.nlp_w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            p=ref_states, 
            lbg=self.lbg, 
            ubg=self.ubg)
        #
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._s_dim:self._s_dim+self._u_dim]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self._s_dim+self._u_dim:2*(self._s_dim+self._u_dim)]) + list(sol_x0[self._s_dim+self._u_dim:])
        
        #
        x0_array = np.reshape(sol_x0[:-self._s_dim], newshape=(-1, self._s_dim+self._u_dim))
        
        # return optimal action, and a sequence of predicted optimal trajectory.  
        return opt_u, x0_array
    
    def sys_dynamics(self, dt):
        M = 4       # refinement
        DT = dt/M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            k1 =DT*self.f(X, U)
            k2 =DT*self.f(X+0.5*k1, U)
            k3 =DT*self.f(X+0.5*k2, U)
            k4 =DT*self.f(X+k3, U)
            #
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6        
        # Fold
        F = ca.Function('F', [X0, U], [X])
        return F
            