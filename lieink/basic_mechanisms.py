from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np
from beartype import beartype

from lieink.annotations import NDArray_6_N, NDArray_N_6_1
from lieink.containers import Container, LieContainer
from lieink.utils import batch_func


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
        pjoint_num = joint_types.count("P")  # type: ignore
        if pjoint_num > 3:
            raise ValueError(
                f"The maximum number of passive joints is 3, got {pjoint_num}"
            )

        self.kinematic_parameters = kinematic_parameters
        self.joint_types = list(joint_types)
        self.joint_num = joint_num

    @property
    def kinematic_parameters(self):
        return self._kinematic_parameters

    @kinematic_parameters.setter
    def kinematic_parameters(self, value: Any):
        self._kinematic_parameters = value

    @abstractmethod
    def forward_kinematics(self, ctrl: Any) -> Any:
        raise NotImplementedError

    def forward_kinematicsb(self, ctrls: Any, *args, **kwargs) -> Any:
        return batch_func(self.forward_kinematics, ctrls, *args, **kwargs)


@beartype
class BasicLimb(BasicSerialMechanism, ABC):
    def __init__(
        self,
        kinematic_parameters: NDArray_N_6_1 | NDArray_6_N | LieContainer,
        joint_types: Iterable[str],
        joint_actuation_types: Iterable[str],
    ):

        for joint_actuation_type in joint_actuation_types:
            if joint_actuation_type not in ["UMP", "MP", "A"]:
                raise ValueError(
                    f"Unsupported joint actuation type: {joint_actuation_type}"
                )
        if len(joint_actuation_types) != len(joint_types):  # type: ignore
            raise ValueError(
                f"Joint actuation types length mismatch: expected {len(joint_types)}, got {len(joint_actuation_types)}"  # type: ignore
            )
        super().__init__(kinematic_parameters, joint_types)
        self.joint_actuation_types = list(joint_actuation_types)
        self.mask_passive_joints = np.array(
            [
                "P" in joint_actuation_type
                for joint_actuation_type in joint_actuation_types
            ],
            dtype=bool,
        )
        self.mask_measurable_joints = np.array(
            [
                "U" not in joint_actuation_type
                for joint_actuation_type in joint_actuation_types
            ],
            dtype=bool,
        )
        self.mask_actuated_joints = np.array(
            [
                "A" in joint_actuation_type
                for joint_actuation_type in joint_actuation_types
            ],
            dtype=bool,
        )
        self.mask_unmeasurable_joints = np.logical_not(self.mask_measurable_joints)


@beartype
class BasicParallelMechanism[T: BasicLimb](ABC):
    def __init__(
        self,
        limb_type: type[T],
        kinematic_parameters: Iterable[Any],
        joint_types: Iterable[Iterable[str]],
        joint_actuation_types: Iterable[Iterable[str]],
    ):
        limb_num = len(kinematic_parameters)  # type: ignore
        if len(joint_types) != limb_num or len(joint_actuation_types) != limb_num:  # type: ignore
            raise ValueError(
                f"Joint types and joint actuation types length mismatch: expected {limb_num}, got {len(joint_types)} and {len(joint_actuation_types)}"  # type: ignore
            )

        limb_container = Container(limb_type)
        for i in range(limb_num):
            limb = limb_type(  # type: ignore
                kinematic_parameters[i],  # type: ignore
                joint_types[i],  # type: ignore
                joint_actuation_types[i],  # type: ignore
            )
            limb_container.append(limb)  # type: ignore

        self.limbs = limb_container

    @property
    def limbs(self):
        return self._limbs

    @limbs.setter
    def limbs(self, value: Container[T] | list[T]):

        if not isinstance(value, Container):
            value = Container[T](value)

        limb_num = len(value)
        count = 0
        joint_num = sum(limb.joint_num for limb in value)
        mask_passive_joints = np.zeros(joint_num, dtype=bool)
        mask_measurable_joints = np.zeros(joint_num, dtype=bool)
        mask_actuated_joints = np.zeros(joint_num, dtype=bool)

        for i in range(limb_num):
            limb = value[i]
            mask_passive_joints[count : count + limb.joint_num] = (
                limb.mask_passive_joints
            )
            mask_measurable_joints[count : count + limb.joint_num] = (
                limb.mask_measurable_joints
            )
            mask_actuated_joints[count : count + limb.joint_num] = (
                limb.mask_actuated_joints
            )
            count += limb.joint_num

        if "limb_num" in self.__dict__:
            if self.limb_num != limb_num:
                raise ValueError(
                    f"Limb number mismatch: expected {self.limb_num}, got {limb_num}"
                )
            elif self.mask_measurable_joints != mask_measurable_joints:
                raise ValueError(
                    f"Measurable joint mask mismatch: expected {self.mask_measurable_joints}, got {mask_measurable_joints}"
                )
            elif self.mask_passive_joints != mask_passive_joints:
                raise ValueError(
                    f"Passive joint mask mismatch: expected {self.mask_passive_joints}, got {mask_passive_joints}"
                )
            elif self.mask_actuated_joints != mask_actuated_joints:
                raise ValueError(
                    f"Actuated joint mask mismatch: expected {self.mask_actuated_joints}, got {mask_actuated_joints}"
                )
            elif self.joint_num != joint_num:
                raise ValueError(
                    f"Joint number mismatch: expected {self.joint_num}, got {joint_num}"
                )

        else:
            self._limbs = value
            self.limb_num = limb_num
            self.mask_passive_joints = mask_passive_joints
            self.mask_measurable_joints = mask_measurable_joints
            self.mask_actuated_joints = mask_actuated_joints
            self.joint_num = joint_num

    @abstractmethod
    def inverse_kinematics(
        self, pose: Any, only_actuated_joints: bool = True, *args, **kwargs
    ) -> Any:
        raise NotImplementedError

    def inverse_kinematicsb(
        self,
        poses: Any,
        only_actuated_joints: bool = True,
        *args,
        **kwargs,
    ) -> Any:
        return batch_func(
            self.inverse_kinematics, poses, only_actuated_joints, *args, **kwargs
        )
