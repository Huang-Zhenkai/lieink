from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from beartype import beartype

from lieink.annotations import NDArray_6_N, NDArray_N_6_1, RealScalar
from lieink.containers import Container, LieContainer
from lieink.utils import ATOL


def batch_calculator(func: Callable, key_paramters, *args, **kwargs):
    key_paramters = np.asarray(key_paramters)

    data_num = key_paramters.shape[0]
    result_first = func(key_paramters[0], *args, **kwargs)
    if isinstance(result_first, tuple):
        results = [[result_first[i]] for i in range(len(result_first))]
        for i in range(1, data_num):
            result = func(key_paramters[i], *args, **kwargs)
            for j in range(len(result_first)):
                results[j].append(result[j])
        return tuple(results)
    else:
        results = [result_first]
        for i in range(1, data_num):
            result = func(key_paramters[i], *args, **kwargs)
            results.append(result)
        return results


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

    @abstractmethod
    def forward_kinematics(self, control_parameter: Any) -> Any:
        raise NotImplementedError

    def forward_kinematicsb(self, control_parameters: Any, *args, **kwargs) -> Any:
        return batch_calculator(
            self.forward_kinematics, control_parameters, *args, **kwargs
        )


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

        joint_num = sum(1 for sub in joint_types for _ in sub)
        mask_passive_joints = np.zeros(joint_num, dtype=bool)
        mask_measurable_joints = np.zeros(joint_num, dtype=bool)
        mask_actuated_joints = np.zeros(joint_num, dtype=bool)
        count = 0

        limb_container = Container(limb_type)
        for i in range(limb_num):
            limb = limb_type(  # type: ignore
                kinematic_parameters[i],  # type: ignore
                joint_types[i],  # type: ignore
                joint_actuation_types[i],  # type: ignore
            )
            limb_container.append(limb)  # type: ignore
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

        self.limbs = limb_container
        self.limb_num = limb_num
        self.mask_passive_joints = mask_passive_joints
        self.mask_measurable_joints = mask_measurable_joints
        self.mask_actuated_joints = mask_actuated_joints
        self.joint_num = joint_num

    def inverse_kinematics_ideal(
        self, pose: Any, only_actuated_joints: bool = True, *args, **kwargs
    ) -> Any:
        raise NotImplementedError

    def inverse_kinematics_idealb(
        self,
        poses: Any,
        only_actuated_joints: bool = True,
        *args,
        **kwargs,
    ) -> Any:
        return batch_calculator(
            self.inverse_kinematics_ideal, poses, only_actuated_joints, *args, **kwargs
        )

    def coordinate_pose_by_ctrl(
        self,
        ctrl: Any,
        return_ctrl: bool = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: bool = False,
        dont_raise: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def coordinate_pose_by_ctrlb(
        self,
        ctrls: Any,
        return_ctrls: bool = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        kwargs["return_NDArray"] = False
        return batch_calculator(
            self.coordinate_pose_by_ctrl,
            ctrls,
            return_ctrls,
            max_iter,
            tol,
            dont_raise,
            *args,
            **kwargs,
        )

    def coordinate_pose(
        self,
        pose: Any,
        return_ctrl: bool = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: bool = False,
        dont_raise: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        ctrl = self.inverse_kinematics_ideal(pose, only_actuated_joints=False)
        return self.coordinate_pose_by_ctrl(
            ctrl,
            return_ctrl,
            max_iter,
            tol,
            return_NDArray,
            dont_raise,
            *args,
            **kwargs,
        )

    def coordinate_poseb(
        self,
        poses: Any,
        return_ctrls: bool = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: bool = False,
        *args,
        **kwargs,
    ) -> Any:

        ctrls = self.inverse_kinematics_idealb(poses, only_actuated_joints=False)
        return self.coordinate_pose_by_ctrlb(
            ctrls, return_ctrls, max_iter, tol, dont_raise, *args, **kwargs
        )
