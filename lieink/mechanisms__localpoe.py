from collections.abc import Iterable
from typing import Literal, overload

import numpy as np

from lieink.annotations import (
    NDArray_1D,
    NDArray_2D,
    NDArray_4_4,
    NDArray_6_N,
    NDArray_N_4_4,
    NDArray_N_6_1,
    NDArray_N_M_4_4,
    NDArray_N_M_6_1,
)
from lieink.atoms import SE3, Twist
from lieink.basic_mechanisms import BasicSerialMechanism
from lieink.containers import LieContainer


class SerialMechanism(BasicSerialMechanism):
    def __init__(
        self,
        kinematic_parameters: NDArray_N_6_1 | NDArray_6_N | LieContainer,
        joint_types: Iterable[str],
    ):

        if isinstance(kinematic_parameters, LieContainer):
            if kinematic_parameters.content_type == Twist:
                kinematic_parameters = kinematic_parameters.toNDArray_3D()
            else:
                raise ValueError(
                    f"Unsupported content type: {kinematic_parameters.content_type}"
                )
        else:
            kinematic_parameters = Twist.reshapeb(kinematic_parameters)

        super().__init__(kinematic_parameters, joint_types)

        kinematic_parameters_SE3 = Twist.expcb(kinematic_parameters, 1, skip_check=True)
        kinematic_parameters_Ad = SE3.toAdcb(kinematic_parameters_SE3, skip_check=True)

        joint_twists = np.zeros((self.jn, 6, 1))
        for i, joint_type in enumerate(self.jts):
            if joint_type == "R":
                joint_twists[i, 2, 0] = 1
            else:
                joint_twists[i, 5, 0] = 1

        self.joint_twists = joint_twists
        self.kinematic_parameters_SE3 = kinematic_parameters_SE3
        self.kinematic_parameters_Ad = kinematic_parameters_Ad

    @overload
    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: Literal[False] = False,
        return_local_poses: Literal[False] = False,
        return_NDArray: Literal[False] = False,
    ) -> SE3: ...

    @overload
    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: Literal[False] = False,
        return_local_poses: Literal[False] = False,
        return_NDArray: Literal[True] = True,
    ) -> NDArray_4_4: ...

    @overload
    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: Literal[False] = False,
        return_local_poses: Literal[True] = True,
        return_NDArray: Literal[False] = False,
    ) -> tuple[SE3, LieContainer]: ...

    @overload
    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: Literal[False] = False,
        return_local_poses: Literal[True] = True,
        return_NDArray: Literal[True] = True,
    ) -> tuple[NDArray_4_4, NDArray_N_4_4]: ...

    @overload
    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: Literal[True] = True,
        return_local_poses: Literal[False] = False,
        return_NDArray: Literal[False] = False,
    ) -> tuple[SE3, LieContainer]: ...

    @overload
    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: Literal[True] = True,
        return_local_poses: Literal[False] = False,
        return_NDArray: Literal[True] = True,
    ) -> tuple[NDArray_4_4, NDArray_N_6_1]: ...

    @overload
    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: Literal[True] = True,
        return_local_poses: Literal[True] = True,
        return_NDArray: Literal[False] = False,
    ) -> tuple[SE3, LieContainer, LieContainer]: ...

    @overload
    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: Literal[True] = True,
        return_local_poses: Literal[True] = True,
        return_NDArray: Literal[True] = True,
    ) -> tuple[NDArray_4_4, NDArray_N_6_1, NDArray_N_4_4]: ...

    def forward_kinematics(
        self,
        control_parameter: NDArray_1D | Iterable,
        return_vjacobian: bool = False,
        return_local_poses: bool = False,
        return_NDArray: bool = False,
    ) -> (
        SE3
        | tuple[SE3, LieContainer]
        | tuple[SE3, LieContainer, LieContainer]
        | NDArray_4_4
        | tuple[NDArray_4_4, NDArray_N_6_1]
        | tuple[NDArray_4_4, NDArray_N_4_4]
        | tuple[NDArray_4_4, NDArray_N_6_1, NDArray_N_4_4]
    ):

        control_parameter = np.asarray(control_parameter)
        if control_parameter.shape[0] != self.jn:
            raise ValueError(
                f"control_parameter must have shape ({self.jn},), but control_parameter.shape[0] is {control_parameter.shape[0]}"
            )
        joint_SE3 = Twist.expcb(self.joint_twists, control_parameter, skip_check=True)
        local_poses = np.zeros((self.jn + 1, 4, 4))
        local_poses[0] = np.eye(4)
        local_poses[1:, :, :] = self.kinematic_parameters_SE3[:-1] @ joint_SE3

        for i in range(1, self.jn):
            local_poses[i + 1] = local_poses[i] @ local_poses[i + 1]
        end_pose: NDArray_4_4 = local_poses[-1] @ self.kinematic_parameters_SE3[-1]

        if return_vjacobian:
            vjacobian = SE3.toAdcb(local_poses[1:, :, :], skip_check=True)
            vjacobian = vjacobian @ self.joint_twists

            match (return_local_poses, return_NDArray):
                case (False, True):
                    return end_pose, vjacobian
                case (True, True):
                    return end_pose, vjacobian, local_poses
                case (False, False):
                    vjacobian_container = LieContainer(Twist)
                    return SE3(end_pose), vjacobian_container.extend(vjacobian)
                case (True, False):
                    vjacobian_container = LieContainer(Twist)
                    local_poses_container = LieContainer(SE3)
                    return (
                        SE3(end_pose),
                        vjacobian_container.extend(vjacobian),
                        local_poses_container.extend(local_poses),
                    )
        else:
            match (return_local_poses, return_NDArray):
                case (False, True):
                    return end_pose
                case (True, True):
                    return end_pose, local_poses
                case (False, False):
                    return SE3(end_pose)
                case (True, False):
                    local_poses_container = LieContainer(SE3)
                    return SE3(end_pose), local_poses_container.extend(local_poses)

    @overload
    def forward_kinematicsb(
        self,
        control_parameters: Iterable[NDArray_1D | Iterable] | NDArray_2D,
        return_vjacobian: Literal[False] = False,
        return_local_poses: Literal[False] = False,
    ) -> NDArray_N_4_4: ...

    @overload
    def forward_kinematicsb(
        self,
        control_parameters: Iterable[NDArray_1D | Iterable] | NDArray_2D,
        return_vjacobian: Literal[True] = True,
        return_local_poses: Literal[False] = False,
    ) -> tuple[NDArray_N_4_4, NDArray_N_M_6_1]: ...

    @overload
    def forward_kinematicsb(
        self,
        control_parameters: Iterable[NDArray_1D | Iterable] | NDArray_2D,
        return_vjacobian: Literal[False] = False,
        return_local_poses: Literal[True] = True,
    ) -> tuple[NDArray_N_4_4, NDArray_N_M_4_4]: ...

    @overload
    def forward_kinematicsb(
        self,
        control_parameters: Iterable[NDArray_1D | Iterable] | NDArray_2D,
        return_vjacobian: Literal[True] = True,
        return_local_poses: Literal[True] = True,
    ) -> tuple[NDArray_N_4_4, NDArray_N_M_6_1, NDArray_N_M_4_4]: ...

    def forward_kinematicsb(
        self,
        control_parameters: Iterable[NDArray_1D | Iterable] | NDArray_2D,
        return_vjacobian: bool = False,
        return_local_poses: bool = False,
    ) -> (
        NDArray_N_4_4
        | tuple[NDArray_N_4_4, NDArray_N_M_6_1]
        | tuple[NDArray_N_4_4, NDArray_N_M_4_4]
        | tuple[NDArray_N_4_4, NDArray_N_M_6_1, NDArray_N_M_4_4]
    ):
        control_parameters = np.asarray(control_parameters)
        if control_parameters.shape[1] != self.jn:
            raise ValueError(
                f"control_parameters must have shape ({self.jn},), but control_parameters.shape[1] is {control_parameters.shape[1]}"
            )

        data_num = control_parameters.shape[0]
        local_poses: NDArray_N_M_4_4 = np.zeros((data_num, self.jn + 1, 4, 4))
        local_poses[:, 0] = np.eye(4)
        for i in range(self.jn):
            joint_SE3 = Twist.expcb(
                np.broadcast_to(self.joint_twists[i], (data_num, 6, 1)),
                control_parameters[:, i],
            )
            local_poses[:, i + 1] = (
                local_poses[:, i] @ self.kinematic_parameters_SE3[i] @ joint_SE3
            )
        end_poses: NDArray_N_4_4 = (
            local_poses[:, -1] @ self.kinematic_parameters_SE3[-1]
        )

        if return_vjacobian:
            vjacobian: NDArray_N_M_6_1 = np.zeros((data_num, self.jn, 6, 1))
            for i in range(self.jn):
                vjacobian[:, i] = (
                    SE3.toAdcb(local_poses[:, i + 1], skip_check=True)
                    @ self.joint_twists[i]
                )
            if return_local_poses:
                return end_poses, vjacobian, local_poses
            else:
                return end_poses, vjacobian
        else:
            if return_local_poses:
                return end_poses, local_poses
            else:
                return end_poses
