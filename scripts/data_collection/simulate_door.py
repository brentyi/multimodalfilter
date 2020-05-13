import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import fannypack
import robosuite
import waypoint_policies
from robosuite.wrappers import IKWrapper

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("target_path", type=str)
    parser.add_argument("--policy", choices=["push", "pull"], required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--visualize_observations", action="store_true")
    parser.add_argument(
        "--traj_count", type=int, default=1, help="Number of trajectories to run."
    )

    args = parser.parse_args()

    # SETTINGS
    preview_mode = args.preview
    vis_images = args.visualize_observations
    target_path = args.target_path
    policy_mode = args.policy
    # /SETTINGS

    if preview_mode:
        vis_images = False

    env = robosuite.make(
        "PandaDoor",
        placement_initializer=True,
        has_renderer=preview_mode,
        ignore_done=True,
        use_camera_obs=(not preview_mode),
        camera_name="birdview",
        camera_height=64,
        camera_width=64,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=20,
        controller="position",
        camera_depth=True,
    )

    # IK controller: we only use this IK, not control
    ik_controller = IKWrapper(env).controller

    trajectories_file = fannypack.utils.TrajectoriesFile(target_path, read_only=False)

    while len(trajectories_file) < args.traj_count:
        obs = env.reset()
        if preview_mode:
            env.render()
        env.controller.step = 0.0
        env.controller.last_goal_position = np.array((0, 0, 0))
        env.controller.last_goal_orientation = np.eye(3)

        # Initialize training policy
        policy: waypoint_policies.AbstractWaypointPolicy
        if policy_mode == "push":
            policy = waypoint_policies.PushWaypointPolicy(env.model.door_offset)
        elif policy_mode == "pull":
            policy = waypoint_policies.PullWaypointPolicy(
                env.model.door_offset, ik_controller
            )
        else:
            assert False

        # Set initial joint and door position
        initial_joints, initial_door = policy.get_initial_state()
        env.set_robot_joint_positions(initial_joints)
        env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("door_hinge")
        ] = initial_door

        q_limit_counter = 0.0

        if vis_images:
            plt.figure()
            plt.gca().invert_yaxis()
            plt.ion()
            plt.show()

        max_iteration_count = 800
        for i in tqdm(range(max_iteration_count)):
            action = policy.update(env)
            # if policy_mode == "pull":
            #     # Open gripper at first few timesteps
            #     # Helps with being a bit more tolerant to initialization issues
            #     action[3] = 1.0
            obs, reward, done, info = env.step(action)
            if preview_mode:
                env.render()

            if env._check_q_limits():
                q_limit_counter += 1.0
                termination_cause = "joint limits"
            elif not obs["contact-obs"]:
                q_limit_counter += 1.0
                termination_cause = "missing contact"
            else:
                q_limit_counter *= 0.9

            if q_limit_counter > 400.0:
                break

            if not args.preview:
                image = np.mean(obs["image"], axis=2) / 127.5 - 1.0
                # image = image[20:20+32,20:20+32]
                obs["image"] = image
                # obs['depth'] = obs['depth'][20:20+32,20:20+32]

                if vis_images:
                    plt.imshow(image, cmap="gray")
                    plt.gca().invert_yaxis()
                    plt.draw()
                    plt.pause(0.0001)

            if type(policy) == waypoint_policies.PushWaypointPolicy:
                if (
                    env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("door_hinge")]
                    < 0.01
                ):
                    termination_cause = "closed door"
                    break

            trajectories_file.add_timestep(obs)

        if i == max_iteration_count - 1:
            termination_cause = "max iteration"
        print(f"Terminated rollout #{len(trajectories_file)}: {termination_cause}")

        if termination_cause == "max iteration" and not args.preview:
            with trajectories_file:
                trajectories_file.complete_trajectory()
        else:
            trajectories_file.abandon_trajectory()
