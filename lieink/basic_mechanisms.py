from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from beartype import beartype


@beartype
class BasicSerialMechanism(ABC):
    def __init__(self, kinematic_parameters: Any, joint_types: Iterable[str]):
        joint_num = len(joint_types)  # type: ignore
        if len(kinematic_parameters) != joint_num + 1:
            raise ValueError(
                f"Kinematic parameters length mismatch: expected {joint_num + 1}, got {len(kinematic_parameters)}"
            )
        for joint_type in joint_types:
            if joint_type not in ["R", "P"]:
                raise ValueError(f"Unsupported joint type: {joint_type}")

        self.kinematic_parameters = kinematic_parameters
        self.joint_types = joint_types
        self.joint_num = joint_num

    @abstractmethod
    def forward_kinematics(self, control_parameter: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward_kinematicsb(self, control_parameters: Any) -> Any:
        raise NotImplementedError

    @property
    def kps(self):
        return self.kinematic_parameters

    @property
    def jts(self):
        return self.joint_types

    @property
    def jn(self):
        return self.joint_num
