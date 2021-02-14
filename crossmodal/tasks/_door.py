import argparse
import sys
from typing import Any, Dict, List

import fannypack
import numpy as np
import torchfilter

from ._task import Task

dataset_urls = {
    "panda_door_pull_10.hdf5": "https://drive.google.com/open?id=1sO3avSEtegDcgISHdALDMW59b-knRRvf",
    "panda_door_pull_100.hdf5": "https://drive.google.com/open?id=1HCDnimAhCDP8OGZimWMRlq8MkrRzOcgf",
    "panda_door_pull_300.hdf5": "https://drive.google.com/open?id=1YSvBR7-JAnH88HRVFAZwiJNY_osLm8aH",
    "panda_door_pull_500.hdf5": "https://drive.google.com/open?id=1dE_jw3-JyX2JagFnCwrfjex4-mwvlEC-",
    "panda_door_push_10.hdf5": "https://drive.google.com/open?id=1nZsQE6FtQwyLkfUQL4CPEc01LjYa_QFy",
    "panda_door_push_100.hdf5": "https://drive.google.com/open?id=1JEDGZWpWE-Z9kuCvRBJh_Auhc-2V0UpN",
    "panda_door_push_300.hdf5": "https://drive.google.com/open?id=18AnusvGEWYA52MHHciq5rHwHJmlx-Ldm",
    "panda_door_push_500.hdf5": "https://drive.google.com/open?id=1TgMp0RIjzxdw6zrRMzGC5tutxYqQ_Tze",
}


class DoorTask(Task):
    """Dataset definition and model registry for door manipulation task."""

    @classmethod
    def add_dataset_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add dataset options to an argument parser.

        Args:
            parser (argparse.ArgumentParser): Parser to add arguments to.
        """
        parser.add_argument("--no_vision", action="store_true")
        parser.add_argument("--no_proprioception", action="store_true")
        parser.add_argument("--no_haptics", action="store_true")
        parser.add_argument("--image_blackout_ratio", type=float, default=0.0)
        parser.add_argument("--sequential_image_rate", type=int, default=1)

    @classmethod
    def get_dataset_args(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """Get a dataset_args dictionary from a parsed set of arguments.

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

    @classmethod
    def get_train_trajectories(
        cls, **dataset_args
    ) -> List[torchfilter.types.TrajectoryNumpy]:
        return _load_trajectories(
            "panda_door_pull_100.hdf5", "panda_door_push_100.hdf5", **dataset_args
        )

    @classmethod
    def get_eval_trajectories(
        cls, **dataset_args
    ) -> List[torchfilter.types.TrajectoryNumpy]:
        return _load_trajectories(
            "panda_door_pull_10.hdf5", "panda_door_push_10.hdf5", **dataset_args
        )


def _load_trajectories(
    *input_files,
    use_vision: bool = True,
    use_proprioception: bool = True,
    use_haptics: bool = True,
    vision_interval: int = 10,
    image_blackout_ratio: float = 0.0,
    sequential_image_rate: int = 1,
    start_timestep: int = 0,
) -> List[torchfilter.types.TrajectoryNumpy]:
    """Loads a list of trajectories from a set of input files, where each trajectory is
    a tuple containing...
        states: an (T, state_dim) array of state vectors
        observations: a key->(T, *) dict of observations
        controls: an (T, control_dim) array of control vectors

    Each input can either be a string or a (string, int) tuple, where int indicates the
    maximum number of trajectories to import.


    Args:
        *input_files: Trajectory inputs. Should be members of
            `crossmodal.door_data.dataset_urls.keys()`.

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
        List[torchfilter.types.TrajectoryNumpy]: list of trajectories.
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

            # The object-state key is stored as:
            # (contact, door theta, door velocity, hinge x, hinge y)
            #
            # We want:
            # (door theta, hinge x, hinge y)
            state_dim = 3
            states = np.full((timesteps, state_dim), np.nan, dtype=np.float32)

            states[:, 0] = raw_trajectory["object-state"][:, 1]  # theta
            states[:, 1] = raw_trajectory["object-state"][:, 3]  # theta
            states[:, 2] = raw_trajectory["object-state"][:, 4]  # theta

            # Pull out observations
            ## This is currently consisted of:
            ## > gripper_pos: end effector position
            ## > gripper_sensors: F/T, contact sensors
            ## > image: camera image

            observations = {}
            observations["gripper_pos"] = raw_trajectory["eef_pos"]
            assert observations["gripper_pos"].shape == (timesteps, 3)

            observations["gripper_sensors"] = np.concatenate(
                (
                    raw_trajectory["ee-force-obs"],
                    raw_trajectory["ee-torque-obs"],
                    raw_trajectory["contact-obs"][:, np.newaxis],
                ),
                axis=1,
            )
            assert observations["gripper_sensors"].shape[1] == 7

            # Zero out proprioception or haptics if unused
            if not use_proprioception:
                observations["gripper_pos"][:] = 0
            if not use_haptics:
                observations["gripper_sensors"][:] = 0

            # Resize images, depth
            raw_trajectory["image"] = raw_trajectory["image"][:, ::2, ::2]
            # raw_trajectory["depth"] = raw_trajectory["depth"][:, ::2, ::2]

            observations["image"] = raw_trajectory["image"]
            assert observations["image"].shape == (timesteps, 32, 32)

            # Mask image observations based on dataset args
            image_mask: np.ndarray
            if not use_vision:
                # Use the zero mask
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

            # observations["depth"] = np.zeros_like(raw_trajectory["depth"])
            # if use_depth:
            #     for i in range(len(observations["depth"])):
            #         index = (i // vision_interval) * vision_interval
            #         index = min(index, len(observations["depth"]))
            #         observations["depth"][i] = raw_trajectory["depth"][index]

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
                [[0.37334135, -0.10821614, 1.5769919]],
                dtype=np.float32,
            )
            observations["gripper_pos"] /= np.array(
                [[0.13496609, 0.14862472, 0.04533212]],
                dtype=np.float32,
            )
            observations["gripper_sensors"] -= np.array(
                [
                    [
                        11.064128,
                        -1.7103539,
                        28.303621,
                        0.06923943,
                        1.661722,
                        -0.14174654,
                        0.63155425,
                    ]
                ],
                dtype=np.float32,
            )
            observations["gripper_sensors"] /= np.array(
                [
                    [
                        36.36674,
                        18.355747,
                        58.651367,
                        1.8596123,
                        4.574878,
                        0.64844555,
                        0.48232532,
                    ]
                ],
                dtype=np.float32,
            )
            states -= np.array(
                [[0.64900873, -0.00079839, -0.00069189]],
                dtype=np.float32,
            )
            states /= np.array(
                [[0.39479038, 0.05650279, 0.0565098]],
                dtype=np.float32,
            )
            controls -= np.array(
                [
                    [
                        3.7333974e-01,
                        -1.0831217e-01,
                        1.5769361e00,
                        3.1821314e-06,
                        9.5862495e-05,
                        4.8311016e-05,
                        6.3155425e-01,
                    ]
                ],
                dtype=np.float32,
            )
            controls /= np.array(
                [
                    [
                        0.134951,
                        0.14904341,
                        0.04531819,
                        0.00323106,
                        0.00411722,
                        0.00165688,
                        0.48232532,
                    ]
                ],
                dtype=np.float32,
            )

            trajectories.append(
                torchfilter.types.TrajectoryNumpy(
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
    """Helper for producing code to normalize inputs"""
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
