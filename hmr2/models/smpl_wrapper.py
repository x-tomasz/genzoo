import torch
import numpy as np
import pickle
from typing import Optional
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput


class SMPL(smplx.SMPLLayer):
    NUM_JOINTS = NUM_BODY_JOINTS = 34
    SHAPE_SPACE_DIM = 145

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertex_joint_selector.extra_joints_idxs = torch.empty(0, dtype=torch.int32)
