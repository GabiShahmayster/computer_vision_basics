import numpy as np
from datetime import datetime

radiansToDegrees = 1 / np.pi * 180.0
degreesToRadians = 1 / 180.0 * np.pi
circumferenceToRadius: float = 1 / 2 / np.pi
KilometersPerHourToMetersPerSec = 1 / 3.6
MetersPerSecToKilometersPerHour = 3.6
MilesPerHourToMetersPerSec = 1.609344 * 1000 / 3600
MicrosecondsToSeconds: float = 1 / 1000000.0
MillisecondsToSeconds: float = 1 / 1000.0
MillimetersToMeters: float = 1 / 1000.0
inchesToMeters: float = 0.0254

EarthRotationRate = 7.292115e-5
SemiMajorAxis = 6378137.0
Flattening = 1 / 298.257223563
Eccentricity = 0.0818191908426215
Mu = 3986004.415e8
J2 = 1.0826269e-3
J3 = -2.5323e-6
gravityStandard = 9.80665

secondsInWeek = 24*7*3600
secondsInDay = 24*3600
secondsInHour = 3600
GpsTimeEpoch = datetime(1980,1,6,0,0,0,0)
