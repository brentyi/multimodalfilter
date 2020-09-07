import abc
import argparse
from typing import Dict

import torchfilter


class Task(abc.ABC):
    model_types: Dict[str, type] = {}
    Filter: type = None

    def __init__(self, *unused_args, **unused_kwargs):
        assert False, "Cannot instantiate task object"

    def __init_subclass__(cls, **kwargs):
        """Registers a task.
        """
        super().__init_subclass__(**kwargs)

        # Each task will have a list of model types.
        cls.model_types = {}

        # Each model type can be registered by subclassing task.Filter.
        class Filter:
            def __init_subclass__(cls_inner: torchfilter.base.Filter, **kwargs):
                assert issubclass(cls_inner, torchfilter.base.Filter)
                cls.model_types[cls_inner.__name__] = cls_inner

        cls.Filter = Filter

    @classmethod
    @abc.abstractclassmethod
    def add_dataset_arguments(cls, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractclassmethod
    def get_dataset_args(cls, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractclassmethod
    def get_train_trajectories(cls):
        pass

    @classmethod
    @abc.abstractclassmethod
    def get_eval_trajectories(cls):
        pass


class PushTaskKloss(Task):
    @classmethod
    def add_dataset_arguments(cls, *args, **kwargs):
        _push_data.add_dataset_arguments(*args, **kwargs)

    @classmethod
    def get_dataset_args(cls, *args, **kwargs):
        pass

    @classmethod
    def get_train_trajectories(cls):
        pass

    @classmethod
    def get_eval_trajectories(cls):
        pass

    pass


class DoorTask(Task):
    pass
