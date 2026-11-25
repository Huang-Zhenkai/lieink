from abc import ABC
from collections.abc import Iterable
from typing import Literal, overload

import numpy as np
import scipy.linalg as slg
from beartype import beartype

from lieink.annotations import (
    NDArray_1D,
    NDArray_2D,
    NDArray_3_N,
    NDArray_4_4,
    NDArray_4D,
    NDArray_6_N,
    NDArray_N_3_1,
    NDArray_N_3_3,
    NDArray_N_4_4,
    NDArray_N_6_1,
    NDArray_N_M_6_1,
    NDArrayBool_1D,
    RealScalar,
)
from lieink.atoms import SE3, SO3, Point3, Twist, Vector3, Wrench
from lieink.basic_mechanisms import BasicLimb, BasicParallelMechanism
from lieink.containers import LieContainer
from lieink.mechanisms.localpoe.serial_mechanisms import SerialMechanism
from lieink.utils import (
    ATOL,
    NDArray_3Dto2D,
    NDArray_4Dto3D,
)


@beartype
class Limb(BasicLimb, SerialMechanism):
    @overload
    def constraint_wrenches(
        self,
        vjacobian: NDArray_N_6_1 | LieContainer,
        return_NDArray: Literal[False] = False,
    ) -> LieContainer: ...

    @overload
    def constraint_wrenches(
        self, vjacobian: NDArray_N_6_1 | LieContainer, return_NDArray: Literal[True]
    ) -> NDArray_N_6_1: ...

    def constraint_wrenches(
        self, vjacobian: NDArray_N_6_1 | LieContainer, return_NDArray: bool = False
    ) -> NDArray_N_6_1 | LieContainer:
        if isinstance(vjacobian, LieContainer):
            vjacobian = vjacobian.toNDArray_3D()
        vjacobian_2D = vjacobian.squeeze(axis=-1)[self.mask_passive_joints]
        constraint_wrenches = Wrench.reshapeb(slg.null_space(vjacobian_2D))

        if return_NDArray:
            return constraint_wrenches
        else:
            constraint_wrenches_container = LieContainer(Wrench)
            return constraint_wrenches_container.extend(constraint_wrenches)

    def constraint_wrenchesb(self, vjacobian: NDArray_N_M_6_1) -> NDArray_N_M_6_1:

        vjacobian_2D = vjacobian.squeeze(axis=-1)[:, self.mask_passive_joints]
        constraint_wrenches = slg.null_space(vjacobian_2D)

        return constraint_wrenches.transpose(0, 2, 1)[..., None]


@beartype
class ParallelMechanism(BasicParallelMechanism[Limb], ABC):
    def __init__(
        self,
        kinematic_parameters: Iterable[NDArray_N_6_1 | NDArray_6_N | LieContainer],
        joint_types: Iterable[Iterable[str]],
        joint_actuation_types: Iterable[Iterable[str]],
    ):
        if isinstance(kinematic_parameters[0], LieContainer):  # type: ignore
            kinematic_parameters = [
                kinematic_parameter.toNDArray_3D()  # type: ignore
                for kinematic_parameter in kinematic_parameters
            ]
        else:
            kinematic_parameters = [
                Twist.reshapeb(kinematic_parameter)  # type: ignore
                for kinematic_parameter in kinematic_parameters
            ]

        super().__init__(Limb, kinematic_parameters, joint_types, joint_actuation_types)

    @overload
    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[False] = False,
        dont_raise: Literal[False] = False,
    ) -> SE3: ...

    @overload
    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[False] = False,
        dont_raise: Literal[False] = False,
    ) -> tuple[SE3, list[NDArray_1D]]: ...

    @overload
    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[False] = False,
        dont_raise: Literal[True] = True,
    ) -> tuple[SE3, bool]: ...

    @overload
    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[False] = False,
        dont_raise: Literal[True] = True,
    ) -> tuple[SE3, list[NDArray_1D], bool]: ...

    @overload
    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[True] = True,
        dont_raise: Literal[False] = False,
    ) -> NDArray_4_4: ...

    @overload
    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[True] = True,
        dont_raise: Literal[False] = False,
    ) -> tuple[NDArray_4_4, list[NDArray_1D]]: ...

    @overload
    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[True] = True,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_4_4, bool]: ...

    @overload
    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[True] = True,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_4_4, list[NDArray_1D], bool]: ...

    def coordinate_pose_by_ctrl(
        self,
        ctrl: Iterable[Iterable[RealScalar] | NDArray_1D],
        return_ctrl: bool = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: bool = False,
        dont_raise: bool = False,
    ) -> (
        SE3
        | tuple[SE3, list[NDArray_1D]]
        | tuple[SE3, bool]
        | tuple[SE3, list[NDArray_1D], bool]
        | NDArray_4_4
        | tuple[NDArray_4_4, list[NDArray_1D]]
        | tuple[NDArray_4_4, bool]
        | tuple[NDArray_4_4, list[NDArray_1D], bool]
    ):

        ln = self.limb_num
        jn_index = list(np.cumsum([0] + [self.limbs[i].joint_num for i in range(ln)]))
        jn_passive_index = list(
            np.cumsum([0] + [sum(self.limbs[i].mask_passive_joints) for i in range(ln)])
        )

        ctrl = np.array([i for limb_ctrl in ctrl for i in limb_ctrl])

        jacobian_theta = np.zeros((ln * 6 - 6, sum(self.mask_passive_joints)))
        errors = np.zeros((ln * 6 - 6, 1))
        pose_all = np.zeros((ln, 4, 4))
        vjacobian_passive_all = np.zeros((6, jn_passive_index[-1]))
        success = True

        for _ in range(max_iter):
            for j in range(ln):
                ctrl_j = ctrl[jn_index[j] : jn_index[j + 1]]
                pose_j, vjacobian_j = self.limbs[j].forward_kinematics(
                    ctrl_j, return_vjacobian=True, return_NDArray=True
                )
                pose_all[j] = pose_j
                vjacobian_passive_all[
                    :, jn_passive_index[j] : jn_passive_index[j + 1]
                ] = NDArray_3Dto2D(vjacobian_j[self.limbs[j].mask_passive_joints])

                if j > 0:
                    errors[j * 6 - 6 : j * 6] = SE3.logc(
                        pose_all[j] @ SE3.invc(pose_all[j - 1]), "Twist"
                    )
                    jacobian_theta[
                        j * 6 - 6 : j * 6, jn_passive_index[j - 1] : jn_passive_index[j]
                    ] = vjacobian_passive_all[
                        :, jn_passive_index[j - 1] : jn_passive_index[j]
                    ]
                    jacobian_theta[
                        j * 6 - 6 : j * 6, jn_passive_index[j] : jn_passive_index[j + 1]
                    ] = -vjacobian_passive_all[
                        :, jn_passive_index[j] : jn_passive_index[j + 1]
                    ]

            errors_norm = np.linalg.norm(errors)
            if errors_norm < tol:
                break

            delta_ctrl_passive = np.linalg.solve(jacobian_theta, errors)
            ctrl[self.mask_passive_joints] += delta_ctrl_passive.flatten()

        else:
            success = False

        pose_all_twist = SE3.logcb(pose_all, "Twist")
        pose = Twist.expc(pose_all_twist.mean(axis=0), 1)
        if return_ctrl:
            ctrl = [ctrl[jn_index[j] : jn_index[j + 1]] for j in range(ln)]
            match return_NDArray, dont_raise:
                case True, True:
                    return pose, ctrl, success
                case True, False:
                    return pose, ctrl
                case False, True:
                    return SE3(pose), ctrl, success
                case False, False:
                    return SE3(pose), ctrl
        else:
            match return_NDArray, dont_raise:
                case True, True:
                    return pose, success
                case True, False:
                    return pose
                case False, True:
                    return SE3(pose), success
                case False, False:
                    return SE3(pose)

    @overload
    def coordinate_pose_by_ctrlb(
        self,
        ctrls: Iterable[Iterable[Iterable[RealScalar] | NDArray_1D] | NDArray_2D],
        return_ctrls: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: Literal[False] = False,
    ) -> NDArray_N_4_4: ...

    @overload
    def coordinate_pose_by_ctrlb(
        self,
        ctrls: Iterable[Iterable[Iterable[RealScalar] | NDArray_1D] | NDArray_2D],
        return_ctrls: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_N_4_4, NDArrayBool_1D]: ...

    @overload
    def coordinate_pose_by_ctrlb(
        self,
        ctrls: Iterable[Iterable[Iterable[RealScalar] | NDArray_1D] | NDArray_2D],
        return_ctrls: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: Literal[False] = False,
    ) -> tuple[NDArray_N_4_4, list[list[NDArray_1D]]]: ...

    @overload
    def coordinate_pose_by_ctrlb(
        self,
        ctrls: Iterable[Iterable[Iterable[RealScalar] | NDArray_1D] | NDArray_2D],
        return_ctrls: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_N_4_4, list[list[NDArray_1D]], NDArrayBool_1D]: ...

    def coordinate_pose_by_ctrlb(
        self,
        ctrls: Iterable[Iterable[Iterable[RealScalar] | NDArray_1D] | NDArray_2D],
        return_ctrls: bool = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: bool = False,
    ) -> (
        NDArray_N_4_4
        | tuple[NDArray_N_4_4, list[list[NDArray_1D]]]
        | tuple[NDArray_N_4_4, NDArrayBool_1D]
        | tuple[NDArray_N_4_4, list[list[NDArray_1D]], NDArrayBool_1D]
    ):

        ln = self.limb_num
        jn_index = list(np.cumsum([0] + [self.limbs[i].joint_num for i in range(ln)]))
        jn_passive_index = list(
            np.cumsum([0] + [sum(self.limbs[i].mask_passive_joints) for i in range(ln)])
        )

        ctrls = np.array(
            [[i for limb_ctrl in ctrl_i for i in limb_ctrl] for ctrl_i in ctrls],
            dtype=np.float64,
        )
        data_num = len(ctrls)

        jacobian_theta = np.zeros((data_num, ln * 6 - 6, sum(self.mask_passive_joints)))
        errors = np.zeros((data_num, ln * 6 - 6, 1))
        pose_all = np.zeros((data_num, ln, 4, 4))
        vjacobian_passive_all = np.zeros((data_num, 6, jn_passive_index[-1]))
        mask_not_converge = np.ones(data_num, dtype=bool)

        for _ in range(max_iter):
            for j in range(ln):
                ctrl_j = ctrls[mask_not_converge, jn_index[j] : jn_index[j + 1]]
                pose_j, vjacobian_j = self.limbs[j].forward_kinematicsb(
                    ctrl_j, return_vjacobian=True
                )
                pose_all[mask_not_converge, j] = pose_j
                vjacobian_passive_all[
                    mask_not_converge, :, jn_passive_index[j] : jn_passive_index[j + 1]
                ] = (
                    vjacobian_j[:, self.limbs[j].mask_passive_joints]
                    .squeeze(axis=-1)
                    .swapaxes(1, 2)
                )

                if j > 0:
                    errors[mask_not_converge, j * 6 - 6 : j * 6, 0] = SE3.logcb(
                        pose_all[mask_not_converge, j]
                        @ SE3.invcb(pose_all[mask_not_converge, j - 1]),
                        "Twist",
                    ).squeeze(-1)
                    jacobian_theta[
                        mask_not_converge,
                        j * 6 - 6 : j * 6,
                        jn_passive_index[j - 1] : jn_passive_index[j],
                    ] = vjacobian_passive_all[
                        mask_not_converge,
                        :,
                        jn_passive_index[j - 1] : jn_passive_index[j],
                    ]
                    jacobian_theta[
                        mask_not_converge,
                        j * 6 - 6 : j * 6,
                        jn_passive_index[j] : jn_passive_index[j + 1],
                    ] = -vjacobian_passive_all[
                        mask_not_converge,
                        :,
                        jn_passive_index[j] : jn_passive_index[j + 1],
                    ]

            errors_norm = np.linalg.norm(errors, axis=1).flatten()
            mask_errors = errors_norm > tol
            mask_not_converge = np.logical_and(mask_not_converge, mask_errors)
            if np.sum(mask_not_converge) == 0:
                break

            delta_ctrl_passive = np.linalg.solve(
                jacobian_theta[mask_not_converge], errors[mask_not_converge]
            )
            ctrls[np.ix_(mask_not_converge, self.mask_passive_joints)] += (
                delta_ctrl_passive.squeeze(-1)
            )
        else:
            if not dont_raise:
                raise ValueError("Not converge")

        pose_all_twist = SE3.logcb(pose_all[:, 0], "Twist")
        pose = Twist.expcb(pose_all_twist, 1)
        if return_ctrls:
            ctrls_ = [
                [ctrl[jn_index[j] : jn_index[j + 1]] for j in range(ln)]
                for ctrl in ctrls
            ]
            if dont_raise:
                return pose, ctrls_, np.logical_not(mask_not_converge)
            return pose, ctrls_
        else:
            if dont_raise:
                return pose, np.logical_not(mask_not_converge)
            return pose

    @overload
    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[False] = False,
        dont_raise: Literal[False] = False,
    ) -> SE3: ...

    @overload
    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[False] = False,
        dont_raise: Literal[True] = True,
    ) -> tuple[SE3, bool]: ...

    @overload
    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[False] = False,
        dont_raise: Literal[False] = False,
    ) -> tuple[SE3, list[NDArray_1D]]: ...

    @overload
    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[False] = False,
        dont_raise: Literal[True] = True,
    ) -> tuple[SE3, list[NDArray_1D], bool]: ...

    @overload
    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[True] = True,
        dont_raise: Literal[False] = False,
    ) -> NDArray_4_4: ...

    @overload
    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[True] = True,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_4_4, bool]: ...

    @overload
    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[True] = True,
        dont_raise: Literal[False] = False,
    ) -> tuple[NDArray_4_4, list[NDArray_1D]]: ...

    @overload
    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: Literal[True] = True,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_4_4, list[NDArray_1D], bool]: ...

    def coordinate_pose(
        self,
        pose: SE3 | NDArray_4_4,
        return_ctrl: bool = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        return_NDArray: bool = False,
        dont_raise: bool = False,
    ) -> (
        SE3
        | tuple[SE3, list[NDArray_1D]]
        | tuple[SE3, bool]
        | tuple[SE3, list[NDArray_1D], bool]
        | NDArray_4_4
        | tuple[NDArray_4_4, list[NDArray_1D]]
        | tuple[NDArray_4_4, bool]
        | tuple[NDArray_4_4, list[NDArray_1D], bool]
    ):
        ctrl = self.inverse_kinematics(pose, only_actuated_joints=False)
        return self.coordinate_pose_by_ctrl(
            ctrl, return_ctrl, max_iter, tol, return_NDArray, dont_raise
        )

    @overload
    def coordinate_poseb(
        self,
        poses: NDArray_N_4_4,
        return_ctrls: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: Literal[False] = False,
    ) -> NDArray_N_4_4: ...

    @overload
    def coordinate_poseb(
        self,
        poses: NDArray_N_4_4,
        return_ctrls: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: Literal[False] = False,
    ) -> tuple[NDArray_N_4_4, list[list[NDArray_1D]]]: ...

    @overload
    def coordinate_poseb(
        self,
        poses: NDArray_N_4_4,
        return_ctrls: Literal[False] = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_N_4_4, NDArrayBool_1D]: ...

    @overload
    def coordinate_poseb(
        self,
        poses: NDArray_N_4_4,
        return_ctrls: Literal[True] = True,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: Literal[True] = True,
    ) -> tuple[NDArray_N_4_4, list[list[NDArray_1D]], NDArrayBool_1D]: ...

    def coordinate_poseb(
        self,
        poses: NDArray_N_4_4,
        return_ctrls: bool = False,
        max_iter: int = 100,
        tol: RealScalar = ATOL,
        dont_raise: bool = False,
    ) -> (
        NDArray_N_4_4
        | tuple[NDArray_N_4_4, list[list[NDArray_1D]]]
        | tuple[NDArray_N_4_4, NDArrayBool_1D]
        | tuple[NDArray_N_4_4, list[list[NDArray_1D]], NDArrayBool_1D]
    ):
        ctrls = self.inverse_kinematicsb(poses, only_actuated_joints=False)
        return self.coordinate_pose_by_ctrlb(
            ctrls, return_ctrls, max_iter, tol, dont_raise
        )


@beartype
class BasicStewartPlate(ParallelMechanism, ABC):
    def __init__(
        self,
        lower_joint_center_in_global_frame: NDArray_N_3_1
        | NDArray_3_N
        | LieContainer[Point3],
        upper_joint_center_in_global_frame: NDArray_N_3_1
        | NDArray_3_N
        | LieContainer[Point3],
        initial_pose: SE3 | NDArray_4_4,
        z1s_in_global_frame: NDArray_N_3_1 | NDArray_3_N | LieContainer[Vector3],
        z6s_in_global_frame: NDArray_N_3_1 | NDArray_3_N | LieContainer[Vector3],
    ):

        if isinstance(lower_joint_center_in_global_frame, LieContainer):
            ljc_g = lower_joint_center_in_global_frame.toNDArray_3D()  # type: ignore
        else:
            ljc_g = Point3.reshapeb(lower_joint_center_in_global_frame)  # type: ignore
        if isinstance(upper_joint_center_in_global_frame, LieContainer):
            ujc_g = upper_joint_center_in_global_frame.toNDArray_3D()  # type: ignore
        else:
            ujc_g = Point3.reshapeb(upper_joint_center_in_global_frame)  # type: ignore
        if isinstance(initial_pose, SE3):
            initial_pose = initial_pose.v
        else:
            initial_pose = SE3._check_shape_and_value(initial_pose)
        z1s = z1s_in_global_frame
        z6s = z6s_in_global_frame
        if isinstance(z1s, LieContainer):
            z1s = z1s.toNDArray_3D()
        else:
            z1s = Vector3.reshapeb(z1s)
        z1s /= np.linalg.norm(z1s, axis=1, keepdims=True)
        if isinstance(z6s, LieContainer):
            z6s = z6s.toNDArray_3D()
        else:
            z6s = Vector3.reshapeb(z6s)
        z6s /= np.linalg.norm(z6s, axis=1, keepdims=True)
        z6s_e = initial_pose[:3, :3].T @ z6s

        ls = ujc_g - ljc_g
        initial_length = np.linalg.norm(ls, axis=1)
        ls = ls / initial_length.reshape(-1, 1, 1)
        ls_e = initial_pose[:3, :3].T @ ls

        zs = np.zeros((6, 7, 3, 1))
        zs[:, 0] = z1s
        zs[:, 1] = np.cross(z1s.squeeze(-1), ls.squeeze(-1))[..., None]
        zs[:, 1] /= np.linalg.norm(zs[:, 1], axis=1, keepdims=True)
        zs[:, 2] = ls
        zs[:, 3] = ls
        zs[:, 4] = np.cross(ls.squeeze(-1), z6s.squeeze(-1))[..., None]
        zs[:, 4] /= np.linalg.norm(zs[:, 4], axis=1, keepdims=True)
        zs[:, 5] = z6s

        Rs = np.zeros((6, 7, 3, 3))
        for i in range(6):
            if i == 0:
                x, y = self._calculate_rotX_and_rotYb(zs[:, i])
                Rs[:, i] = SO3.rotxcb(x) @ SO3.rotycb(y)
            else:
                x, y = self._calculate_rotX_and_rotYb(
                    SO3.invcb(Rs[:, i - 1]) @ zs[:, i]
                )
                Rs[:, i] = Rs[:, i - 1] @ SO3.rotxcb(x) @ SO3.rotycb(y)

        Rs[:, 6] = initial_pose[:3, :3]

        ts = self.calculate_initial_ts(ljc_g, ujc_g, initial_pose)
        poses = np.zeros((6, 7, 4, 4))
        poses[:, :, -1, -1] = 1.0
        poses[:, :, :3, :3] = Rs
        poses[:, :, :3, -1:] = ts

        kpss = np.zeros((6, 7, 6, 1))
        for i in range(7):
            if i == 0:
                kps = SE3.logcb(poses[:, i], logto="Twist")
            else:
                kps = SE3.logcb(SE3.invcb(poses[:, i - 1]) @ poses[:, i], logto="Twist")
            kpss[:, i] = kps

        if self.config == "SPS":  # type: ignore 这个不做了吧，反正没用上
            joint_type = ["R", "R", "R", "P", "R", "R"]
            joint_actuated = ["UMP", "UMP", "UMP", "A", "UMP", "UMP"]
        else:
            joint_type = ["R", "R", "P", "R", "R", "R"]
            joint_actuated = ["UMP", "UMP", "A", "UMP", "UMP", "UMP"]

        super().__init__(kpss, [joint_type] * 6, [joint_actuated] * 6)

        Rs_ljc = self._calculate_R_by_X_Yb(z1s, zs[:, 1])
        initial_q1, initial_q2 = self._calculate_rotX_and_rotYb(SO3.invcb(Rs_ljc) @ ls)

        Rs_ujc = self._calculate_R_by_X_Yb(
            z6s_e, np.cross(z6s_e.squeeze(-1), ls_e.squeeze(-1)).reshape(-1, 3, 1)
        )
        initial_q6, initial_q5 = self._calculate_rotX_and_rotYb(
            SO3.invcb(Rs_ujc) @ ls_e
        )

        cos_q4 = np.sum(zs[:, 1] * zs[:, 4], axis=1).flatten()
        sin_q4 = np.sum(
            np.cross(zs[:, 1].squeeze(-1), zs[:, 4].squeeze(-1)) * ls.squeeze(-1),
            axis=-1,
        )
        initial_q4 = np.arctan2(sin_q4, cos_q4)

        self.ljc_g = Point3.toPoint4cb(ljc_g)
        self.ujc_e = SE3.invc(initial_pose) @ Point3.toPoint4cb(ujc_g)

        self.z1s_g = z1s
        self.z6s_e = z6s_e

        self.initial_q1s = initial_q1
        self.initial_q2s = initial_q2
        self.initial_q3s = initial_length.flatten()
        self.initial_q4s = initial_q4
        self.initial_q5s = initial_q5
        self.initial_q6s = initial_q6

        self.Rs_ljc_T = SO3.invcb(Rs_ljc)
        self.Rs_ujc_T = SO3.invcb(Rs_ujc)

    def calculate_initial_ts(
        self,
        ljc_g: NDArray_N_3_1,
        ujc_g: NDArray_N_3_1,
        initial_pose: NDArray_4_4,
    ) -> NDArray_4D:

        ts = np.zeros((6, 7, 3, 1))

        ts[:, 0] = ljc_g
        ts[:, 1] = ljc_g
        ts[:, 2] = ljc_g
        ts[:, 4] = ujc_g
        ts[:, 5] = ujc_g
        ts[:, 6] = initial_pose[:3, -1:]
        if self.config == "UPS" or self.config == "SPS":  # type: ignore
            ts[:, 3] = ujc_g
        else:
            ts[:, 3] = ljc_g

        return ts

    def inverse_kinematics(self, pose: NDArray_4_4, only_actuated_joints: bool = True):
        pose = SE3._check_shape_and_value(pose).reshape(-1, 4, 4)
        return self.inverse_kinematicsb(pose, only_actuated_joints)[0]

    def inverse_kinematicsb(
        self, poses: NDArray_N_4_4, only_actuated_joints: bool = True
    ):

        poses = SE3._check_shapeb_and_valueb(poses)

        data_num = poses.shape[0]
        ujc_g = np.einsum("nij,mjk->nmik", poses, self.ujc_e)
        ls = (ujc_g - self.ljc_g)[:, :, :-1]
        length = np.linalg.norm(ls, axis=2).squeeze(-1)

        if only_actuated_joints:
            return length - self.initial_q3s

        ls /= length.reshape(-1, 6, 1, 1)

        qs = np.zeros((data_num, 6, 6))
        qs[:, :, 2] = length - self.initial_q3s

        ls_ljc = self.Rs_ljc_T @ ls
        q1s, q2s = self._calculate_rotX_and_rotYb(NDArray_4Dto3D(ls_ljc))
        qs[:, :, 0] = q1s.reshape(data_num, -1) - self.initial_q1s
        qs[:, :, 1] = q2s.reshape(data_num, -1) - self.initial_q2s

        ls_ujc = self.Rs_ujc_T @ (poses[:, None, :3, :3].swapaxes(-1, -2) @ ls)
        q6s, q5s = self._calculate_rotX_and_rotYb(NDArray_4Dto3D(ls_ujc))
        qs[:, :, 4] = q5s.reshape(data_num, -1) - self.initial_q5s
        qs[:, :, 5] = -(q6s.reshape(data_num, -1) - self.initial_q6s)

        z6s_g = poses[:, None, :3, :3] @ self.z6s_e
        ls_3d = ls.squeeze(-1)

        z2s_g = np.cross(self.z1s_g.squeeze(-1), ls_3d)
        z5s_g = np.cross(ls_3d, z6s_g.squeeze(-1))

        z2s_g /= np.linalg.norm(z2s_g, axis=-1, keepdims=True)
        z5s_g /= np.linalg.norm(z5s_g, axis=-1, keepdims=True)

        cos_q4 = np.sum(z2s_g * z5s_g, axis=-1)
        sin_q4 = np.sum(np.cross(z2s_g, z5s_g) * ls_3d, axis=-1)
        q4 = np.arctan2(sin_q4, cos_q4)

        qs[:, :, 3] = (q4 - self.initial_q4s + np.pi) % (2 * np.pi) - np.pi

        return qs

    def _calculate_rotX_and_rotYb(self, zb):

        norm = np.linalg.norm(zb, axis=1, keepdims=True)
        zb = zb / norm

        zx, zy, zz = zb[:, 0, 0], zb[:, 1, 0], zb[:, 2, 0]
        r = np.hypot(zy, zz)

        x = np.arctan2(zy, -zz)
        y = np.arctan2(zx, -r)

        return x, y

    def _calculate_R_by_X_Yb(
        self, Xs: NDArray_N_3_1, Ys: NDArray_N_3_1
    ) -> NDArray_N_3_3:

        n = Xs.shape[0]
        Rs = np.zeros((n, 3, 3))
        Rs[:, :, 0:1] = Xs
        Rs[:, :, 0] /= np.linalg.norm(Rs[:, :, 0], axis=1, keepdims=True)
        Rs[:, :, 1:2] = Ys
        Rs[:, :, 1] /= np.linalg.norm(Rs[:, :, 1], axis=1, keepdims=True)
        Rs[:, :, 2] = np.cross(Rs[:, :, 0], Rs[:, :, 1])
        Rs[:, :, 2] /= np.linalg.norm(Rs[:, :, 2], axis=1, keepdims=True)

        return Rs


@beartype
class StewartPlate_UCU(BasicStewartPlate):
    config = "UCU"


@beartype
class StewartPlate_UPS(BasicStewartPlate):
    config = "UPS"

    def __init__(
        self,
        lower_joint_center_in_global_frame: NDArray_N_3_1
        | NDArray_3_N
        | LieContainer[Point3],
        upper_joint_center_in_global_frame: NDArray_N_3_1
        | NDArray_3_N
        | LieContainer[Point3],
        initial_pose: SE3 | NDArray_4_4,
        z1s_in_global_frame: NDArray_N_3_1 | NDArray_3_N | LieContainer[Vector3],
    ):

        if isinstance(initial_pose, SE3):
            initial_pose = initial_pose.v
        else:
            initial_pose = SE3._check_shape_and_value(initial_pose)

        z1s = z1s_in_global_frame
        if isinstance(z1s, LieContainer):
            z1s = z1s.toNDArray_3D()
        else:
            z1s = Vector3.reshapeb(z1s)
        z1s /= np.linalg.norm(z1s, axis=1, keepdims=True)
        z6s_e = z1s[[3, 4, 5, 0, 1, 2]]
        z6s = initial_pose[:3, :3] @ z6s_e

        super().__init__(
            lower_joint_center_in_global_frame,
            upper_joint_center_in_global_frame,
            initial_pose,
            z1s,
            z6s,
        )


StewartPlate = StewartPlate_UPS
