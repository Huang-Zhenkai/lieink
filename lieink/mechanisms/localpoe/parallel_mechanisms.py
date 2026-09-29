from collections.abc import Iterable
from typing import Literal, overload

import numpy as np
import scipy.linalg as slg
from beartype import beartype

from lieink.annotations import (
    NDArray_1D,
    NDArray_2D,
    NDArray_4_4,
    NDArray_6_N,
    NDArray_N_4_4,
    NDArray_N_6_1,
    NDArray_N_M_6_1,
    NDArrayBool_1D,
    RealScalar,
)
from lieink.atoms import SE3, SO3, Point3, Twist, Wrench
from lieink.basic_mechanisms import BasicLimb, BasicParallelMechanism
from lieink.containers import LieContainer
from lieink.mechanisms.localpoe.serial_mechanisms import SerialMechanism
from lieink.utils import ATOL, NDArray_3Dto2D, NDArray_3Dto4D, find_zeros


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
class ParallelMechanism(BasicParallelMechanism[Limb]):
    def __init__(
        self,
        kinematic_parameters: Iterable[NDArray_N_6_1 | NDArray_6_N | LieContainer],
        joint_types: Iterable[Iterable[str]],
        joint_actuation_types: Iterable[Iterable[str]],
    ):

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
        return super().coordinate_pose(
            pose, return_ctrl, max_iter, tol, return_NDArray, dont_raise
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
        return super().coordinate_poseb(poses, return_ctrls, max_iter, tol, dont_raise)


@beartype
class StewartPlate(ParallelMechanism):
    def __init__(
        self,
        lower_hinges_in_global_frame: NDArray_N_6_1
        | NDArray_6_N
        | LieContainer[Point3],
        upper_hinges_in_end_frame: NDArray_N_6_1 | NDArray_6_N | LieContainer[Point3],
        initial_pose: SE3 | NDArray_4_4,
    ):
        if isinstance(initial_pose, SE3):
            initial_pose = initial_pose.v
        else:
            initial_pose = SE3._check_shape_and_value(initial_pose)

        lhs_g = np.zeros((6, 4, 1))
        lhs_g[:, -1] = 1
        uhs_e = np.zeros((6, 4, 1))
        uhs_e[:, -1] = 1

        if isinstance(lower_hinges_in_global_frame, LieContainer):
            lower_hinges_in_global_frame = lower_hinges_in_global_frame.toNDArray_3D()
        else:
            lower_hinges_in_global_frame = Point3.reshapeb(lower_hinges_in_global_frame)
        if isinstance(upper_hinges_in_end_frame, LieContainer):
            upper_hinges_in_end_frame = upper_hinges_in_end_frame.toNDArray_3D()
        else:
            upper_hinges_in_end_frame = Point3.reshapeb(upper_hinges_in_end_frame)

        lhs_g[:, :3] = lower_hinges_in_global_frame
        uhs_e[:, :3] = upper_hinges_in_end_frame

        uhs_g = np.einsum("ij,njk->nik", initial_pose, uhs_e)
        initial_ls_vector = (uhs_g - lhs_g)[:, :3]
        initial_ls_length = np.linalg.norm(initial_ls_vector, axis=1)

        kpss = np.zeros((6, 6, 7))

        Zs = np.einsum("nij,nj->nij", initial_ls_vector, 1 / initial_ls_length)

        Xs = np.zeros(Zs.shape)
        Xs[:, 0] = -Zs[:, 1]
        Xs[:, 1] = Zs[:, 0]
        Xs = np.einsum("nij,nj->nij", Xs, 1 / np.linalg.norm(Xs, axis=1))
        Ys = np.cross(Zs.reshape(6, 3), Xs.reshape(6, 3), axis=1).reshape(6, 3, 1)

        S1s = np.zeros((6, 4, 4))
        S1s[:, :3, 0:1] = -Ys
        S1s[:, :3, 1:2] = -Zs
        S1s[:, :3, 2:3] = Xs
        S1s[:, :, 3:] = lhs_g
        rho1 = SE3.logcb(S1s, logto="Twist")
        kpss[:, :, 0:1] = rho1

        S2s = np.zeros((6, 4, 4))
        S2s[:] = SE3.rotyc(-np.pi / 2)
        rho2 = SE3.logcb(S2s, logto="Twist")
        kpss[:, :, 1:2] = rho2

        S3s = np.zeros((6, 4, 4))
        S3s[:] = SE3.rotxc(np.pi / 2)
        rho3 = SE3.logcb(S3s, logto="Twist")
        kpss[:, :, 2:3] = rho3

        S0to3 = np.einsum("nij,njk->nik", S1s, S2s)
        S0to3 = np.einsum("nij,njk->nik", S0to3, S3s)

        S0to4 = np.zeros((6, 4, 4))
        S0to4[:, 1, 0] = -1
        S0to4[:, 2, 1] = -1
        S0to4[:, 0, 2] = 1
        S0to4[:, :, -1:] = uhs_g

        S4 = np.einsum("nij,njk->nik", np.linalg.inv(S0to3), S0to4)
        rho4 = SE3.logcb(S4, logto="Twist")
        kpss[:, :, 3:4] = rho4

        S5 = SE3.rotyc(-np.pi / 2)
        rho5 = SE3.logc(S5, logto="Twist")
        kpss[:, :, 4:5] = rho5

        S6 = SE3.rotxc(np.pi / 2)
        rho6 = SE3.logc(S6, logto="Twist")
        kpss[:, :, 5:6] = rho6

        S0to6 = np.einsum("nij,jk->nik", S0to4, S5)
        S0to6 = np.einsum("nij,jk->nik", S0to6, S6)

        S7 = np.einsum("nij,jk->nik", np.linalg.inv(S0to6), initial_pose)
        rho7 = SE3.logcb(S7, logto="Twist")
        kpss[:, :, 6:7] = rho7
        kpss = [Twist.reshapeb(kps) for kps in kpss]

        super().__init__(
            kpss,
            [["R", "R", "P", "R", "R", "R"]] * 6,
            [["UMP", "UMP", "A", "UMP", "UMP", "UMP"]] * 6,
        )

        self.lhs_g = lhs_g
        self.uhs_e = uhs_e
        self.initial_ls_length = initial_ls_length
        self.initial_ls_vector_norm = Zs

        self._R0to3 = S0to3[:, :3, :3]
        self._R3to0 = np.linalg.inv(S0to3)[:, :3, :3]
        self._R4 = S4[:, :3, :3]

    def inverse_kinematics_ideal(
        self, pose: SE3 | NDArray_4_4, only_actuated_joints: bool = True
    ) -> list[NDArray_1D]:

        if isinstance(pose, SE3):
            pose = pose.v
        else:
            pose = SE3._check_shape_and_value(pose)

        uhs_g = np.einsum("ij,njk->nik", pose, self.uhs_e)
        ls_vector = (uhs_g - self.lhs_g)[:, :-1]
        ls_length = np.linalg.norm(ls_vector, axis=1)
        ls_vector_norm = np.einsum("nij,nj->nij", ls_vector, 1 / ls_length)

        qs = np.zeros((6, 6))
        qs[:, 2:3] = ls_length - self.initial_ls_length

        ls_vector_norm_in_S3 = np.einsum("nij,njk->nik", self._R3to0, ls_vector_norm)

        theta2 = np.arcsin(ls_vector_norm_in_S3[:, 0])
        cos_theta2 = np.cos(theta2)
        theta1 = np.arctan2(
            ls_vector_norm_in_S3[:, 1] / (-cos_theta2),
            ls_vector_norm_in_S3[:, 2] / cos_theta2,
        )

        qs[:, 0:1] = theta1
        qs[:, 1:2] = theta2

        R1 = SO3.rotxcb(theta1.flatten())
        R2 = SO3.rotycb(theta2.flatten())

        R0to3 = np.einsum("nij,njk->nik", self._R0to3, R1)
        R0to3 = np.einsum("nij,njk->nik", R0to3, R2)
        R0to4 = np.einsum("nij,njk->nik", R0to3, self._R4)

        R4to7 = np.einsum("nij,jk->nik", np.linalg.inv(R0to4), pose[:3, :3])

        sin_theta5 = R4to7[:, 2, 2]
        theta5 = np.arcsin(sin_theta5)
        cos_theta5 = np.cos(theta5)

        theta6 = np.zeros(6)
        theta4 = np.zeros(6)

        mask = np.logical_not(find_zeros(cos_theta5))
        cos_theta5 = cos_theta5[mask]
        cos_theta6 = R4to7[mask, 2, 0] / cos_theta5
        sin_theta6 = -R4to7[mask, 2, 1] / cos_theta5
        theta6[mask] = np.arctan2(sin_theta6, cos_theta6)

        cos_theta4 = -R4to7[:, 1, 2] / cos_theta5
        sin_theta4 = R4to7[:, 0, 2] / cos_theta5
        theta4[mask] = np.arctan2(sin_theta4, cos_theta4)

        qs[:, 3] = theta4
        qs[:, 4] = theta5
        qs[:, 5] = theta6

        if only_actuated_joints:
            qs = qs[:, 2:3]

        return [q for q in qs]

    def inverse_kinematics_idealb(
        self, pose: NDArray_N_4_4, only_actuated_joints: bool = True
    ) -> list[list[NDArray_1D]]:

        pose = SE3._check_shapeb_and_valueb(pose)
        data_num = pose.shape[0]

        uhs_g = np.einsum("nij,mjk->nmik", pose, self.uhs_e)
        ls_vector = (uhs_g - self.lhs_g)[:, :, :-1]
        ls_length = np.linalg.norm(ls_vector, axis=2)
        ls_vector_norm = np.einsum("nmij,nmj->nmij", ls_vector, 1 / ls_length)

        qs = np.zeros((data_num, 6, 6))
        qs[:, :, 2:3] = ls_length - self.initial_ls_length

        ls_vector_norm_in_S3 = np.einsum("mij,nmjk->nmik", self._R3to0, ls_vector_norm)

        theta2 = np.arcsin(ls_vector_norm_in_S3[:, :, 0])
        cos_theta2 = np.cos(theta2)
        theta1 = np.arctan2(
            ls_vector_norm_in_S3[:, :, 1] / (-cos_theta2),
            ls_vector_norm_in_S3[:, :, 2] / cos_theta2,
        )

        qs[:, :, 0:1] = theta1
        qs[:, :, 1:2] = theta2

        R1 = NDArray_3Dto4D(SO3.rotxcb(theta1.flatten()), shape=(data_num, -1))
        R2 = NDArray_3Dto4D(SO3.rotycb(theta2.flatten()), shape=(data_num, -1))

        R0to3 = np.einsum("mij,nmjk->nmik", self._R0to3, R1)
        R0to3 = np.einsum("nmij,nmjk->nmik", R0to3, R2)
        R0to4 = np.einsum("nmij,mjk->nmik", R0to3, self._R4)

        R4to7 = np.einsum("nmij,njk->nmik", np.linalg.inv(R0to4), pose[:, :3, :3])

        sin_theta5 = R4to7[:, :, 2, 2]
        theta5 = np.arcsin(sin_theta5)
        cos_theta5 = np.cos(theta5)

        theta6 = np.zeros((data_num, 6))
        theta4 = np.zeros((data_num, 6))

        mask = np.logical_not(find_zeros(cos_theta5))
        cos_theta5 = cos_theta5[mask]
        cos_theta6 = R4to7[mask, 2, 0] / cos_theta5
        sin_theta6 = -R4to7[mask, 2, 1] / cos_theta5
        theta6[mask] = np.arctan2(sin_theta6, cos_theta6)

        cos_theta4 = -R4to7[:, :, 1, 2].flatten() / cos_theta5
        sin_theta4 = R4to7[:, :, 0, 2].flatten() / cos_theta5
        theta4[mask] = np.arctan2(sin_theta4, cos_theta4)

        qs[:, :, 3] = theta4
        qs[:, :, 4] = theta5
        qs[:, :, 5] = theta6

        if only_actuated_joints:
            qs = qs[:, :, 2:3]

        return [[q_i for q_i in q] for q in qs]
