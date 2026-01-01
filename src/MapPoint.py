import json
import numpy as np
from typing import Optional

class MapPointDebug:
    """
    This class contains aux debug information for map points
    """
    simulated_landmark_id: object

    def __init__(self, simulated_landmark_id):
        self.simulated_landmark_id = simulated_landmark_id

    def __str__(self):
        return "id = {0:d}".format(self.simulated_landmark_id)

class MapPoint:
    x: float
    y: float
    z: float
    debug: Optional[MapPointDebug]

    def __init__(self, x: float,
                 y: float,
                 z: float,
                 debug: MapPointDebug = None):
        self.x = x
        self.y = y
        self.z = z
        self.debug = debug

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __str__(self) -> str:
        return json.dumps({"x":self.x,
                           "y":self.y,
                           "z":self.z,
                           "debug":str(self.debug) if self.debug is not None else ""})