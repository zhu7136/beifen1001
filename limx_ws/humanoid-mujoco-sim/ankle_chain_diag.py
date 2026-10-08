import atexit
import csv
import time
import numpy as np
import mujoco


class AnkleChainDiag:
    def __init__(self, model, path="ankle_chain_diag.csv"):
        self.model = model
        self.shadow = mujoco.MjData(model)

        self.motors = [
            "left_A_achilles_joint", "left_B_achilles_joint",
            "right_A_achilles_joint", "right_B_achilles_joint",
        ]
        self.connects = [
            "A_achilles_connect_L", "B_achilles_connect_L",
            "A_achilles_connect_R", "B_achilles_connect_R",
        ]
        self.qadr, self.vadr, self.eqids = [], [], []

        for name in self.motors:
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Missing/non-hinge motor joint: {name}")
            self.qadr.append(int(model.jnt_qposadr[jid]))
            self.vadr.append(int(model.jnt_dofadr[jid]))

        for name in self.connects:
            eid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, name
            )
            if eid < 0 or model.eq_type[eid] != mujoco.mjtEq.mjEQ_CONNECT:
                raise ValueError(f"Missing/non-connect equality: {name}")
            if model.eq_obj1id[eid] < 0 or model.eq_obj2id[eid] < 0:
                raise ValueError(f"Expected two body anchors: {name}")
            self.eqids.append(eid)

        self.file = open(path, "w", newline="", buffering=1)
        atexit.register(self.file.close)
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["wall_time_s", "monotonic_time_s", "sim_time_s"]
            + [f"{n}_qpos_rad" for n in self.motors]
            + [f"{n}_qvel_rad_s" for n in self.motors]
            + [f"{n}_closure_mm" for n in self.connects]
            + [f"{n}_active" for n in self.connects]
        )
        print(f"[CHAIN-DIAG] ready: {path}", flush=True)

    def write(self, data, wall_time):
        mono = time.perf_counter()
        s, m = self.shadow, self.model

        # 只在独立的数据副本中重算几何，不推进物理仿真。
        s.qpos[:] = data.qpos
        s.mocap_pos[:] = data.mocap_pos
        s.mocap_quat[:] = data.mocap_quat
        mujoco.mj_kinematics(m, s)

        errors = []
        for eid in self.eqids:
            b1 = int(m.eq_obj1id[eid])
            b2 = int(m.eq_obj2id[eid])

            # 使用模型编译后的两端局部 anchor，
            # 不是简单相减两个 body 原点。
            a1 = m.eq_data[eid, :3]
            a2 = m.eq_data[eid, 3:6]
            p1 = s.xpos[b1] + s.xmat[b1].reshape(3, 3) @ a1
            p2 = s.xpos[b2] + s.xmat[b2].reshape(3, 3) @ a2
            errors.append(float(np.linalg.norm(p1 - p2) * 1000.0))

        self.writer.writerow(
            [wall_time, mono, float(data.time)]
            + [float(data.qpos[a]) for a in self.qadr]
            + [float(data.qvel[a]) for a in self.vadr]
            + errors
            + [int(data.eq_active[e]) for e in self.eqids]
        )
