#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from fvcore.common.config import CfgNode

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.DATASET_TYPE = CfgNode()
    _C.DATASET_TYPE.CKG = True
    _C.DATASET_TYPE.CKF = True
    _C.DATASET_TYPE.TST = True
    _C.DATASET_TYPE.SYN_EASY = True
    _C.DATASET_TYPE.SYN_HARD = False