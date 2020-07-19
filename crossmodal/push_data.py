import argparse
import sys
from typing import List

import numpy as np

import diffbayes
import fannypack

dataset_urls = {
    "gentle_push_10.hdf5": "https://drive.google.com/file/d/1qmBCfsAGu8eew-CQFmV1svodl9VJa6fX/view?usp=sharing",
    "gentle_push_100.hdf5": "https://drive.google.com/file/d/1PmqQy5myNXSei56upMy3mXKu5Lk7Fr_g/view?usp=sharing",
    "gentle_push_300.hdf5": "https://drive.google.com/file/d/18dr1z0N__yFiP_DAKxy-Hs9Vy_AsaW6Q/view?usp=sharing",
    "gentle_push_1000.hdf5": "https://drive.google.com/file/d/1JTgmq1KPRK9HYi8BgvljKg5MPqT_N4cR/view?usp=sharing",
}


def add_dataset_arguments(parser: argparse.ArgumentParser):
    """Add dataset options to an argument parser.

    Args:
        parser (argparse.ArgumentParser): Parser to add arguments to.
    """
    parser.add_argument("--no_vision", action="store_true")
    parser.add_argument("--no_proprioception", action="store_true")
    parser.add_argument("--no_haptics", action="store_true")
    parser.add_argument("--image_blackout_ratio", type=float, default=0.0)
    parser.add_argument("--sequential_image_rate", type=int, default=1)


def get_dataset_args(args: argparse.Namespace):
    """Get a dataset_args dictionary from a parsed set of arguments, for use in
    `load_trajectories()`.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    dataset_args = {
        "use_vision": not args.no_vision,
        "use_proprioception": not args.no_proprioception,
        "use_haptics": not args.no_haptics,
        "image_blackout_ratio": args.image_blackout_ratio,
        "sequential_image_rate": args.sequential_image_rate,
    }
    return dataset_args


def load_trajectories(
    *input_files,
    use_vision: bool = True,
    use_proprioception: bool = True,
    use_haptics: bool = True,
    vision_interval: int = 10,
    image_blackout_ratio: float = 0.0,
    sequential_image_rate: int = 1,
    start_timestep: int = 0,
) -> List[diffbayes.types.TrajectoryTupleNumpy]:
    """Loads a list of trajectories from a set of input files, where each trajectory is
    a tuple containing...
        states: an (T, state_dim) array of state vectors
        observations: a key->(T, *) dict of observations
        controls: an (T, control_dim) array of control vectors

    Each input can either be a string or a (string, int) tuple, where int indicates the
    maximum number of trajectories to import.


    Args:
        *input_files: Trajectory inputs. Should be members of
            `crossmodal.push_data.dataset_urls.keys()`.

    Keyword Args:
        use_vision (bool, optional): Set to False to zero out camera inputs.
        vision_interval (int, optional): # of times each camera image is duplicated. For
            emulating a slow image rate.
        use_proprioception (bool, optional): Set to False to zero out kinematics data.
        use_haptics (bool, optional): Set to False to zero out F/T sensors.
        image_blackout_ratio (float, optional): Dropout probabiliity for camera inputs.
            0.0 = no dropout, 1.0 = all images dropped out.
        sequential_image_rate (int, optional): If value is `N`, we only send 1 image
            frame ever `N` timesteps. All others are zeroed out.
        start_timestep (int, optional): If value is `N`, we skip the first `N` timesteps
            of each trajectory.

    Returns:
        List[diffbayes.types.TrajectoryTupleNumpy]: list of trajectories.
    """
    trajectories = []

    assert 1 > image_blackout_ratio >= 0
    assert image_blackout_ratio == 0 or sequential_image_rate == 1

    for name in input_files:
        max_trajectory_count = sys.maxsize
        if type(name) == tuple:
            name, max_trajectory_count = name
        assert type(max_trajectory_count) == int

        # Load trajectories file into memory, all at once
        with fannypack.data.TrajectoriesFile(
            fannypack.data.cached_drive_file(name, dataset_urls[name])
        ) as f:
            raw_trajectories = list(f)

        # Iterate over each trajectory
        for raw_trajectory_index, raw_trajectory in enumerate(raw_trajectories):
            if raw_trajectory_index >= max_trajectory_count:
                break

            timesteps = len(raw_trajectory["object-state"])

            # State is just (x, y)
            state_dim = 2
            states = np.full((timesteps, state_dim), np.nan)
            states[:, :2] = raw_trajectory["Cylinder0_pos"][:, :2]  # x, y

            # Pull out observations
            ## This is currently consisted of:
            ## > gripper_pos: end effector position
            ## > gripper_sensors: F/T, contact sensors
            ## > image: camera image

            observations = {}
            observations["gripper_pos"] = raw_trajectory["eef_pos"]
            assert observations["gripper_pos"].shape == (timesteps, 3)

            observations["gripper_sensors"] = np.concatenate(
                (raw_trajectory["force"], raw_trajectory["contact"][:, np.newaxis],),
                axis=1,
            )
            assert observations["gripper_sensors"].shape[1] == 7

            # Zero out proprioception or haptics if unused
            if not use_proprioception:
                observations["gripper_pos"][:] = 0
            if not use_haptics:
                observations["gripper_sensors"][:] = 0

            # Get image
            observations["image"] = raw_trajectory["image"].copy()
            assert observations["image"].shape == (timesteps, 32, 32)

            # Mask image observations based on dataset args
            image_mask: np.ndarray
            if not use_vision:
                # Use a zero mask
                image_mask = np.zeros((timesteps, 1, 1), dtype=np.float32)
            elif image_blackout_ratio == 0.0:
                # Apply sequential rate
                image_mask = np.zeros((timesteps, 1, 1), dtype=np.float32)
                image_mask[::sequential_image_rate, 0, 0] = 1.0
            else:
                # Apply blackout rate
                image_mask = (
                    (np.random.uniform(size=(timesteps,)) > image_blackout_ratio)
                    .astype(np.float32)
                    .reshape((timesteps, 1, 1))
                )
            observations["image"] *= image_mask

            # Pull out controls
            ## This is currently consisted of:
            ## > previous end effector position
            ## > end effector position delta
            ## > binary contact reading
            eef_positions = raw_trajectory["eef_pos"]
            eef_positions_shifted = np.roll(eef_positions, shift=1, axis=0)
            eef_positions_shifted[0] = eef_positions[0]
            controls = np.concatenate(
                [
                    eef_positions_shifted,
                    eef_positions - eef_positions_shifted,
                    raw_trajectory["contact-obs"][:, np.newaxis],
                ],
                axis=1,
            )
            assert controls.shape == (timesteps, 7)

            # Normalize data
            observations["gripper_pos"] -= np.array(
                [[0.46806443, -0.0017836, 0.88028437]], dtype=np.float32
            )
            observations["gripper_pos"] /= np.array(
                [[0.02410769, 0.02341035, 0.04018243]], dtype=np.float32
            )
            observations["gripper_sensors"] -= np.array(
                [
                    [
                        4.9182904e-01,
                        4.5039989e-02,
                        -3.2791464e00,
                        -3.3874984e-03,
                        1.1552566e-02,
                        -8.4817986e-04,
                        2.1303751e-01,
                    ]
                ],
                dtype=np.float32,
            )
            observations["gripper_sensors"] /= np.array(
                [
                    [
                        1.6152629,
                        1.666905,
                        1.9186896,
                        0.14219016,
                        0.14232528,
                        0.01675198,
                        0.40950698,
                    ]
                ],
                dtype=np.float32,
            )
            states -= np.array([[0.4970164, -0.00916641]])
            states /= np.array([[0.0572766, 0.06118315]])
            controls -= np.array(
                [
                    [
                        4.6594709e-01,
                        -2.5247163e-03,
                        8.8094306e-01,
                        1.2939950e-04,
                        -5.4364675e-05,
                        -6.1112235e-04,
                        2.2041667e-01,
                    ]
                ],
                dtype=np.float32,
            )
            controls /= np.array(
                [
                    [
                        0.02239027,
                        0.02356066,
                        0.0405312,
                        0.00054858,
                        0.0005754,
                        0.00046352,
                        0.41451886,
                    ]
                ],
                dtype=np.float32,
            )

            trajectories.append(
                (
                    states[start_timestep:],
                    fannypack.utils.SliceWrapper(observations)[start_timestep:],
                    controls[start_timestep:],
                )
            )

            # Reduce memory usage
            raw_trajectories[raw_trajectory_index] = None
            del raw_trajectory

    ## Uncomment this line to generate the lines required to normalize data
    # _print_normalization(trajectories)

    return trajectories


def _print_normalization(trajectories):
    """ Helper for producing code to normalize inputs
    """
    states = []
    observations = fannypack.utils.SliceWrapper({})
    controls = []
    for t in trajectories:
        states.extend(t[0])
        observations.append(t[1])
        controls.extend(t[2])
    observations = observations.map(
        lambda list_value: np.concatenate(list_value, axis=0)
    )
    print(observations["gripper_sensors"].shape)

    def print_ranges(**kwargs):
        for k, v in kwargs.items():
            mean = repr(np.mean(v, axis=0, keepdims=True))
            stddev = repr(np.std(v, axis=0, keepdims=True))
            print(f"{k} -= np.{mean}")
            print(f"{k} /= np.{stddev}")

    print_ranges(
        gripper_pos=observations["gripper_pos"],
        gripper_sensors=observations["gripper_sensors"],
        states=states,
        controls=controls,
    )


### Reference
#
# Valid trajectory keys:
# [
#     "contact-obs",
#     "depth",
#     "ee-force-obs",
#     "ee-torque-obs",
#     "eef_pos",
#     "eef_quat",
#     "eef_vang",
#     "eef_vlin",
#     "gripper_qpos",
#     "gripper_qvel",
#     "image",
#     "joint_pos",
#     "joint_vel",
#     "object-state",
#     "prev-act",
#     "robot-state",
# ]
