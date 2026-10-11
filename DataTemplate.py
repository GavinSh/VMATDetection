from copy import deepcopy
from enum import Enum
from typing import List

import numpy as np

FIXED_PRESCRIPTION_DOSE = 200


class DeepCopyClassBase:
    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        copy_instance = cls.__new__(cls)
        memodict[id(self)] = copy_instance
        for k, v in self.__dict__.items():
            setattr(copy_instance, k, deepcopy(v, memodict))
        return copy_instance


class TreatmentData(DeepCopyClassBase):
    def __init__(self):
        # treatment ID is unique
        self.treatment_ID: int
        self.patient_ID: str
        self.patient_name: str
        self.treatment_type: RadiotherapyType
        self.treatment_room: TreatmentRoom

        self.organ_code: str
        self.organ_name: str
        # treatment name is unique within patient
        self.treatment_name: str

        self.num_leaf: int = 0
        self.num_arc: int = 0
        self.arcs: List[Arc] = []

    def __len__(self):
        return len(self.arcs)


class SubField(DeepCopyClassBase):
    def __init__(self):
        self.dose: float
        # Subfield dose equivalent to a fixed prescription dose
        self.equivalent_dose: float
        # the positions of the leafs, whose shape is [N, 2]
        self.leaf_position: np.ndarray
        self.coll_angle: float


class Field(DeepCopyClassBase):
    def __init__(self):
        self.gantry_angle: float
        self.gantry_idx: int
        self.gantry_rotation_direction: FieldRotationDirection
        self.num_sub_field: int = 0
        self.sub_fields: List[SubField] = []

    def __len__(self):
        return len(self.sub_fields)


class Arc(DeepCopyClassBase):
    def __init__(self):
        self.arc_name: str
        self.prescription_dose: float
        self.num_field: int = 0
        self.fields: List[Field] = []

    def __len__(self):
        return len(self.fields)


class RadiotherapyType(Enum):
    VMAT = 1
    IMRT = 2
    TOMO = 3


class FieldRotationDirection(Enum):
    """
    龙门架旋转方向
    """
    Clockwise = 1
    Counterclockwise = 2
    Unknown = 3


class TreatmentRoom(Enum):
    """
    加速器室与机头型号
    """
    R1_Agility = 1
    R1_IFNT = 2
    R2_VersaHD = 3
    R3_Synergy = 4
    R4_Novalis = 5
    R5_Unique = 6
    R8_VersaHD = 7
