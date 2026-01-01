from datetime import datetime, timedelta
from numpy import float64
import numpy as np

from src.Constants import secondsInWeek, secondsInDay, GpsTimeEpoch

nat_double_value: float = 9999
nat_datetime_value: datetime = datetime(9999, 1, 1, 0, 0, 0)


class Time:
    def __init__(self, value):
        self.is_nat = False
        if type(value) is float or type(value) is float64:
            self.is_double = True
            self.value_double = value
        elif type(value) is datetime:
            self.is_double = False
            self.value_datetime = value
        elif type(value) is str:
            isot_date_components = value.split('T')[0].split('-')
            isot_time_components = value.split('T')[1].split(':')
            round_seconds = int(float(isot_time_components[2]))
            micro_seconds = int(1e6 * (float(isot_time_components[2]) - round_seconds))
            self.value_datetime = datetime(year=int(isot_date_components[0]),
                                           month=int(isot_date_components[1]),
                                           day=int(isot_date_components[2]),
                                           hour=int(isot_time_components[0]),
                                           minute=int(isot_time_components[1]),
                                           second=round_seconds,
                                           microsecond=micro_seconds)
            self.is_double = False
        else:
            pass

    def get_time_difference(self, target_time: 'Time') -> float:
        assert (type(target_time) is Time), "target_time is not Time object"
        if self.is_nat or target_time.is_nat:
            return nat_double_value
        if self.is_double:
            return self.value_double - target_time.value_double
        else:
            return (self.value_datetime - target_time.value_datetime).total_seconds()

    def shift_by(self, shift_period_sec: float) -> 'Time':
        if self.is_double:
            return Time(self.get_value() + shift_period_sec)
        else:
            return Time(self.get_value() + timedelta(hours=0, minutes=0, seconds=shift_period_sec, microseconds=0))

    def __copy__(self):
        """
        This method returns a copy of the object
        @return:
        """
        return self.shift_by(.0)

    def get_type(self):
        if self.is_double:
            return '[sec]'
        else:
            return '[YYYY-MM-DDTHH:MM:SS.SSS]'

    def get_value(self):
        if self.is_double:
            return self.value_double
        else:
            return self.value_datetime

    def get_value_str(self):
        if self.is_double:
            return str(self.value_double)
        else:
            return self.value_datetime.isoformat()

    def __str__(self):
        if self.is_double:
            return str(self.value_double) + " " + self.get_type()
        else:
            return str(self.value_datetime) + " " + self.get_type()

    def get_time_of_week(self):
        if self.is_double:
            return self.value_double
        else:
            time = self.value_datetime.time()
            day_of_week = self.value_datetime.date().isoweekday()
            if day_of_week == 7:
                # GPS week begins on Sunday midnight
                day_of_week = 0
            return day_of_week * 24 * 3600 + timedelta(hours=time.hour, minutes=time.minute, seconds=time.second,
                                                       microseconds=time.microsecond).total_seconds()


def get_utc_datetime_from_utc_week_and_seconds_of_week_sinch_gps_epoch(utc_week: int,
                                                                       utc_seconds_of_week: float):
    delta_time_total_seconds = utc_week * secondsInWeek + \
                               utc_seconds_of_week
    delta_time_days = np.floor(delta_time_total_seconds / secondsInDay)
    delta_time_seconds_in_day = delta_time_total_seconds - delta_time_days * secondsInDay
    # delta_time_fractional_seconds = delta_time_seconds_in_day % 1
    # delta_time_fractional_milliseconds = round(delta_time_fractional_seconds / MillisecondsToSeconds)

    return GpsTimeEpoch + timedelta(days=delta_time_days,
                                    seconds=delta_time_seconds_in_day)
