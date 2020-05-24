import numpy as np
import torch
from tqdm.auto import tqdm

from fannypack import utils


def load_trajectories(
    *paths,
    use_vision=True,
    vision_interval=10,
    use_proprioception=True,
    use_haptics=True,
    use_mass=False,
    use_depth=False,
    image_blackout_ratio=0,
    sequential_image_rate=1,
    start_timestep=0,
    **unused,
):
    """
    Loads a list of trajectories from a set of input paths, where each
    trajectory is a tuple containing...
        states: an (T, state_dim) array of state vectors
        observations: a key->(T, *) dict of observations
        controls: an (T, control_dim) array of control vectors

    Each path can either be a string or a (string, int) tuple, where int
    indicates the maximum number of trajectories to import.
    """
    trajectories = []

    assert 1 > image_blackout_ratio >= 0
    assert image_blackout_ratio == 0 or sequential_image_rate == 1

    for path in paths:
        count = np.float("inf")
        if type(path) == tuple:
            path, count = path
            assert type(count) == int

        with utils.TrajectoriesFile(path) as f:
            # Keys:
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

            # Iterate over each trajectory
            for i, trajectory in tqdm(enumerate(f)):
                if i >= count:
                    break

                timesteps = len(trajectory["object-state"])

                # The object-state key is stored as:
                # (contact, door theta, door velocity, hinge x, hinge y)
                #
                # We want:
                # (door theta, hinge x, hinge y)
                state_dim = 3
                states = np.full((timesteps, state_dim), np.nan)

                states[:, 0] = trajectory["object-state"][:, 1]  # theta
                states[:, 1] = trajectory["object-state"][:, 3]  # theta
                states[:, 2] = trajectory["object-state"][:, 4]  # theta

                # Pull out observations
                ## This is currently consisted of:
                ## > gripper_pos: end effector position
                ## > gripper_sensors: F/T, contact sensors
                ## > image: camera image

                observations = {}
                observations["gripper_pos"] = trajectory["eef_pos"]
                assert observations["gripper_pos"].shape == (timesteps, 3)

                observations["gripper_sensors"] = np.concatenate(
                    (
                        trajectory["ee-force-obs"],
                        trajectory["ee-torque-obs"],
                        trajectory["contact-obs"][:, np.newaxis],
                    ),
                    axis=1,
                )
                assert observations["gripper_sensors"].shape[1] == 7

                # If disabled, zero out proprioception or haptics
                if not use_proprioception:
                    observations["gripper_pos"][:] = 0
                if not use_haptics:
                    observations["gripper_sensors"][:] = 0

                # Resize images, depth
                trajectory["image"] = trajectory["image"][:, ::2, ::2]
                trajectory["depth"] = trajectory["depth"][:, ::2, ::2]

                assert trajectory["image"].shape[1:] == (32, 32)
                observations["image"] = np.zeros_like(trajectory["image"])
                if use_vision:
                    for i in range(len(observations["image"])):
                        index = (i // vision_interval) * vision_interval
                        index = min(index, len(observations["image"]))
                        blackout_chance = np.random.uniform()
                        # if blackout chance > ratio, then fill image
                        # otherwise zero
                        if image_blackout_ratio == 0 and i % sequential_image_rate == 0:
                            observations["image"][i] = trajectory["image"][index]

                        if (
                            blackout_chance > image_blackout_ratio
                            and sequential_image_rate == 1
                        ):
                            observations["image"][i] = trajectory["image"][index]

                observations["depth"] = np.zeros_like(trajectory["depth"])
                if use_depth:
                    for i in range(len(observations["depth"])):
                        index = (i // vision_interval) * vision_interval
                        index = min(index, len(observations["depth"]))
                        observations["depth"][i] = trajectory["depth"][index]

                # Pull out controls
                ## This is currently consisted of:
                ## > previous end effector position
                ## > end effector position delta
                ## > binary contact reading
                eef_positions = trajectory["eef_pos"]
                eef_positions_shifted = np.roll(eef_positions, shift=-1)
                eef_positions_shifted[-1] = eef_positions[-1]
                controls = np.concatenate(
                    [
                        eef_positions_shifted,
                        eef_positions - eef_positions_shifted,
                        trajectory["contact-obs"][:, np.newaxis],
                    ],
                    axis=1,
                )
                assert controls.shape == (timesteps, 7)

                # Normalize data
                observations["gripper_pos"] -= np.array(
                    [[0.37334135, -0.10821614, 1.5769919]]
                )
                observations["gripper_pos"] /= np.array(
                    [[0.13496609, 0.14862472, 0.04533212]]
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
                    ]
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
                    ]
                )
                states -= np.array([[0.64900873, -0.00079839, -0.00069189]])
                states /= np.array([[0.39479038, 0.05650279, 0.0565098]])
                controls -= np.array(
                    [
                        [
                            -0.1074974,
                            1.5747472,
                            0.374877,
                            0.48083794,
                            -1.6832547,
                            1.2024931,
                            0.63155425,
                        ]
                    ]
                )
                controls /= np.array(
                    [
                        [
                            0.15000738,
                            0.07635409,
                            0.14191878,
                            0.15354143,
                            0.1670589,
                            0.1371131,
                            0.48232532,
                        ]
                    ]
                )

                trajectories.append(
                    (
                        states[start_timestep:],
                        utils.SliceWrapper(observations)[start_timestep:],
                        controls[start_timestep:],
                    )
                )

    ## Uncomment this line to generate the lines required to normalize data
    # _print_normalization(trajectories)

    return trajectories


def _print_normalization(trajectories):
    """ Helper for producing code to normalize inputs
    """
    states = []
    observations = None
    controls = []
    for t in trajectories:
        states.extend(t[0])
        if observations is None:
            observations = utils.SliceWrapper(t[1]).map(lambda x: [x])
        else:
            observations.append(t[1])
        controls.extend(t[2])
    observations = observations.map(lambda x: np.concatenate(x, axis=0)).data
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
