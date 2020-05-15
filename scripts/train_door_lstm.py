"""
"""

import fannypack

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

# dataset_file = "door_pull_10.hdf5"
for dataset_file in dataset_urls.keys():
    dataset_path = fannypack.data.cached_drive_file(
        dataset_file, dataset_urls[dataset_file]
    )

print(dataset_path)
trajectories = fannypack.data.TrajectoriesFile(dataset_path)
