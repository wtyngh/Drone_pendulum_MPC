import os
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
from functools import partial

#
from high_mpc.simulation.dynamic_gap_pend import DynamicGap
from high_mpc.mpc.MPC_pend import MPC
from high_mpc.simulation.animation import SimVisual
#
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video', type=bool, default=False,
        help="Save the animation as a video file")
    return parser

def run_mpc(sim_visual ,env):
    #
    env.reset()
    t, n = 0, 0
    t0 = time.time()
    info_list=[]

    while t < env.sim_T:
        t = env.sim_dt * n
        _, _, _, info = env.step()
        t_now = time.time()
        print(t_now - t0)
	    #
        t0 = time.time()
        #
        n += 1
        update = False

        info_list.append(info)


        sim_visual.updateinfo(t, info)
        plt.pause(0.0001)

def main():
    #
    args = arg_parser().parse_args()
    #
    plan_T = 2.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.1 # Sampling time step for MPC and local planner
    so_path = "./mpc/saved/mpc_v1.so" # saved mpc model (casadi code generation)
    #
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(mpc, plan_T, plan_dt)

    #
    sim_visual = SimVisual(env)

    
    #
    info_list = run_mpc(sim_visual, env)

    #***** plot the animation ***
    # run_frame = partial(run_mpc, env)
    # ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
    #         init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)

    # *** test area ***
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                            frames=200, interval=20, blit=True)
    
    # if args.save_video:
    #     writer = animation.writers["ffmpeg"]
    #     writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    #     ani.save("MPC_0.mp4", writer=writer)
    
    # plt.tight_layout()
    # plt.show()
    
if __name__ == "__main__":
    main()
