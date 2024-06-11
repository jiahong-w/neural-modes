from dataclasses import dataclass
from enum import Enum

class EMatModel(Enum):
    MM_StVKMod = 0 
    MM_NH = 1

@dataclass
class SOMaterial:
    # Bending coefficient
    kB: float = 0
    # Stretching coefficients
    k1: float = 0
    k2: float = 0
    k3: float = 0
    k4: float = 0
    # Damping coefficients
    kD: float = 0
    # mass density
    rho: float = 0
    # material model
    matModel: EMatModel = EMatModel.MM_StVKMod
