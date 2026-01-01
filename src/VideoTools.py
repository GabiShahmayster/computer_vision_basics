from cv2 import FONT_HERSHEY_COMPLEX, FONT_ITALIC, QT_FONT_NORMAL, putText
import numpy as np
from typing import List, Tuple
from collections import namedtuple
from enum import Enum

TextDataAndFormat = namedtuple('TextLineData', ['text', 'color', 'font_scale', 'thickness'])


class ColorTuple(Enum):
    # https://flaviocopes.com/rgb-color-codes/
    red= (0,0,255)
    green= (0,255,0)
    blue= (255,0,0)
    brown= (42,42,165)
    orange= (0,165,255)
    yellow= (0,255,255)
    cyan= (255,255,0)
    gray= (128,128,128)
    white= (255,255,255)
    black= (0,0,0)

class VideoTextOverlay:
    default_font: int
    org: Tuple
    default_font_scale: float
    default_color: ColorTuple
    default_thickness: int

    output_text_buffer: List[TextDataAndFormat]

    is_disabled: bool

    def __init__(self,
                 org: Tuple = (50, 50),
                 default_font_scale: float = 0.45,
                 default_color: ColorTuple = ColorTuple.white,
                 default_thickness: int = 1,
                 default_font: int = FONT_ITALIC):

        self.is_disabled = False
        self.org = org
        self.default_font_scale = default_font_scale
        self.default_color = default_color
        self.default_thickness = default_thickness
        self.default_font = default_font
        self.clear_text_buffer()

    def disable(self):
        """
        This method will effectively disable the write_to_frame() method
        @return:
        """
        self.is_disabled = True

    def enable(self):
        """
        This method will effectively disable the write_to_frame() method
        @return:
        """
        self.is_disabled = False

    def clear_text_buffer(self):
        self.output_text_buffer = list()

    def append_new_line_to_text_buffer(self, line: str,
                                       color: ColorTuple = None,
                                       font_scale: float = None,
                                       thickness: int = None):
        if color is None:
            color = self.default_color
        if font_scale is None:
            font_scale = self.default_font_scale
        if thickness is None:
            thickness = self.default_thickness
        self.output_text_buffer.append(TextDataAndFormat(text=line,
                                                         color=color,
                                                         font_scale=font_scale,
                                                         thickness=thickness))

    def write_to_frame(self, frame: np.ndarray):
        if self.is_disabled:
            # do nothing
            return frame

        for line_idx, line_data in enumerate(self.output_text_buffer):
            line = line_data.text
            if line is None:
                continue
            text_x_coord = self.org[0]
            text_y_coord = self.org[1] + line_idx * 20
            putText(img=frame,
                    text=line,
                    org=(text_x_coord, text_y_coord),
                    fontFace=self.default_font,
                    fontScale=line_data.font_scale,
                    color=line_data.color.value,
                    thickness=line_data.thickness)

    def update_text_buffer_line(self, line_number: int,
                                line: str,
                                color: ColorTuple = None,
                                font_scale: float = None,
                                thickness: int = None):
        if color is None:
            color = self.default_color
        if font_scale is None:
            font_scale = self.default_font_scale
        if thickness is None:
            thickness = self.default_thickness
        self.output_text_buffer[line_number] = TextDataAndFormat(text=line,
                                                                 color=color,
                                                                 font_scale=font_scale,
                                                                 thickness=thickness)


