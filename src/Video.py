import glob
import os
from datetime import datetime
from itertools import count
from json import dumps
from typing import Optional, Union, Generator, Tuple, Dict
from timeit import default_timer as timer

import numpy as np
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES, \
    imshow, waitKey, destroyAllWindows, BORDER_CONSTANT, copyMakeBorder
from easygui import enterbox
from src.TimeUtils import Time
from src.VideoTools import VideoTextOverlay, ColorTuple


class FrameInformation:
    # The captured frame index
    frame_number: int
    # The captured frame
    raw_frame: np.ndarray
    # The captured frame timestamp
    time: Time
    # The output frame
    frame: np.ndarray

    def __init__(self, frame_number: int,
                 raw_frame: np.ndarray,
                 time: Time,
                 frame: np.ndarray):
        self.frame_number = frame_number
        self.raw_frame = raw_frame
        self.time = time
        self.frame = frame

    @classmethod
    def build(cls,
              frame_number: int,
              raw_frame: np.ndarray,
              time: Time):
        return FrameInformation(frame_number,
                                raw_frame,
                                time,
                                np.copy(raw_frame))

    def __copy__(self) -> 'FrameInformation':
        """
        This method returns a copy of the object
        @return:
        """
        return FrameInformation.build(frame_number=self.frame_number,
                                      raw_frame=self.raw_frame,
                                      time=self.time)

    def get_keys_as_dictionary(self) -> Dict:
        """
        This method returns the parameters of this object, as a dictionary
        @return:
        """
        return {"time " + self.time.get_type(): self.time.get_value(),
                "time of week [sec]": self.time.get_time_of_week(),
                "frame_number": self.frame_number}

    def __str__(self):
        """
        This method returns a string representation of the object
        @return:
        """
        return dumps(self.get_keys_as_dictionary(), default=str)


class MockVideoCapture:
    """
    This class bypasses VideoCapture to allow working with a real-time sequence of frames
    mocks VideoCapture's methods called by VideoStreamReader
    """
    frame_width: int
    frame_height: int
    frame_generator: Generator
    current_frame_number: int

    def __init__(self, frame_width: int,
                 frame_height: int,
                 frame_generator: Generator):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_generator = frame_generator
        self.current_frame_number = 0

    def read(self) -> Tuple[bool, np.ndarray]:
        """
        mock VideoCapture.read()
        @return:
        """
        try:
            frame = next(self.frame_generator)
            if frame is not None:
                self.current_frame_number += 1
                return True, frame
            else:
                return False, np.empty([])
        except:
            return False, np.empty([])

    def isOpened(self) -> bool:
        """
        mock VideoCapture.isOpened()
        @return:
        """
        return True

    def release(self):
        """
        mock VideoCapture.release()
        @return:
        """
        return

    def get(self, code: int) -> int:
        """
        mock VideoCapture.get()
        @return:
        """
        if code == CAP_PROP_FRAME_WIDTH:
            # get frame width
            return self.frame_width
        elif code == CAP_PROP_FRAME_HEIGHT:
            # get frame height
            return self.frame_height
        elif code == CAP_PROP_FRAME_COUNT:
            # get number of frames
            return -1
        elif code == CAP_PROP_POS_FRAMES:
            # get current frame number
            return self.current_frame_number

    def set(self, code: int, frame: int):
        """
        mock VideoCapture.set()
        @return:
        """
        return


class VideoStreamReader:
    video_capture: Union[VideoCapture, MockVideoCapture]
    device: Union[int, str]
    frames_per_second: int
    current_frame_number: int
    frame_counter: Optional[object]

    frame_width: int
    frame_height: int
    frame_count: int

    number_of_frames_to_capture: int
    number_of_frames_from_start_to_skip: int
    start_of_capture_timestamp: datetime

    # video playback related functionality
    pause_after_current_frame: bool
    video_skip_to_frame_character: int = ord('m')
    video_pause_character: int = ord(' ')
    video_quit_character: int = ord('q')
    video_next_frame_character: int = ord('n')
    video_previous_frame_character: int = ord('b')
    video_increase_speed_character: int = ord('+')
    video_decrease_speed_character: int = ord('-')

    video_speed_factor: float

    frame_number_overlay: VideoTextOverlay

    BOTTOM_TEXT_OVERLAY_AREA_HEIGHT_PX = 200

    handle_user_input: bool
    add_frame_overlay_area: bool

    fps_timer: Optional[timer]

    def __init__(self, device: Union[int, str, MockVideoCapture] = 0,
                 frames_per_second: int = 30,
                 capture_period_sec: float = None,
                 number_of_frames_from_start_to_skip: int = None):
        self.video_speed_factor = 1.0
        self.device = device
        self.frames_per_second = frames_per_second
        self.pause_after_current_frame = False
        self.handle_user_input = True
        self.add_frame_overlay_area = True

        # Create a VideoCapture object
        if type(device) is str and os.path.isdir(device):  # make sure that it's a folder and not a video file
            # look for an ordered sequence of files, 'img_0000.png','img_0001.png',...
            path_to_first_frame: str = self.get_first_frame_path_formatted_for_video_capture(device)
            self.video_capture = VideoCapture(path_to_first_frame)
        elif type(device) is MockVideoCapture:
            self.video_capture = device
        else:
            self.video_capture = VideoCapture(device)

        if capture_period_sec is None:
            self.number_of_frames_to_capture = np.inf
        else:
            self.number_of_frames_to_capture = int(capture_period_sec * self.frames_per_second)

        if number_of_frames_from_start_to_skip is None:
            self.number_of_frames_from_start_to_skip = 0
        else:
            self.number_of_frames_from_start_to_skip = number_of_frames_from_start_to_skip

        # Check if camera opened successfully
        assert (self.video_capture.isOpened()), "Unable to open video stream"

        # get stream width and height
        self.frame_width = int(self.video_capture.get(CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(CAP_PROP_FRAME_HEIGHT))

        try:
            self.frame_count = int(self.video_capture.get(CAP_PROP_FRAME_COUNT))
            self.frame_counter = None
        except:
            self.frame_counter = count(start=1)
        self.update_current_frame_number()

        if self.number_of_frames_from_start_to_skip > 0:
            self.skip_to_frame(target_frame=self.number_of_frames_from_start_to_skip)

        self.frame_number_overlay = VideoTextOverlay(org=(self.frame_width - 150,
                                                          self.frame_height + 20),
                                                     default_color=ColorTuple.white)
        self.frame_number_overlay.append_new_line_to_text_buffer(line='')
        self.frame_number_overlay.append_new_line_to_text_buffer(line='')

        self.fps_timer = None

    def get_first_frame_path_formatted_for_video_capture(self, path_of_images_directory: str):
        """
        This method returns a string, formatted to allow OpenCV's VideoCapture to read sequentially numbered images
        from a folder
        @param path_of_images_directory:
        @return:
        """
        path_of_first_frame: str = sorted(glob.glob(os.path.join(path_of_images_directory, '*.png')))[0]
        filename_of_first_frame: str = os.path.split(path_of_first_frame)[1]
        index_size: int = len(filename_of_first_frame.split('.png')[0])
        return os.path.join(path_of_images_directory, '%' + str(index_size) + 'd.png')

    def disable_frame_overlay(self):
        """
        This method disables adding frame-information box below the original frame
        @return:
        """
        self.add_frame_overlay_area = False

    def disable_user_input_handling(self):
        """
        This method aborts listening to keyboard
        @return:
        """
        self.handle_user_input = False

    def enable_user_input_handling(self):
        """
        This method aborts listening to keyboard
        @return:
        """
        self.handle_user_input = True

    def init_capture(self):
        self.start_of_capture_timestamp = datetime.now()

    def __iter__(self):
        # self.init_capture()
        return self

    def __next__(self) -> Optional[FrameInformation]:
        if self.handle_user_input:
            self.handle_user_input_events()

        frame = self.get_frame()

        if self.current_frame_number > (
                self.number_of_frames_to_capture + self.number_of_frames_from_start_to_skip):
            self.stop_capture()

        if frame is None:
            if type(self.device) is not MockVideoCapture:
                # could not grab new frame, stopping capture (unless it's a MockVideoCapture)
                self.stop_capture()
            else:
                return None
        else:
            self.update_current_frame_number()
            frame_information: FrameInformation = FrameInformation.build(frame_number=self.current_frame_number,
                                                                         raw_frame=frame,
                                                                         time=Time(datetime.now()))
            if self.add_frame_overlay_area:
                frame_information = self.add_text_overlay_area_to_bottom_of_frame(frame_information=frame_information)
                frame_information = self.overlay_frame_index_window(frame_information=frame_information)
                # if (datetime.now() - self.start_of_capture_timestamp).total_seconds()>self.capture_period_sec:
                #     quit()
            return frame_information

    def add_text_overlay_area_to_bottom_of_frame(self, frame_information: FrameInformation) -> FrameInformation:
        frame_information.frame = copyMakeBorder(src=frame_information.frame,
                                                 top=0, bottom=self.BOTTOM_TEXT_OVERLAY_AREA_HEIGHT_PX, left=0, right=0,
                                                 borderType=BORDER_CONSTANT,
                                                 value=ColorTuple.black.value)
        return frame_information

    def handle_user_input_events(self):
        if self.pause_after_current_frame:
            while True:
                key = waitKey(0) & 0xFF
                if key == self.video_next_frame_character:
                    # wait for next event
                    break
                elif key == self.video_previous_frame_character:
                    # wait for next key
                    self.skip_to_frame(target_frame=self.current_frame_number - 1)
                    break
                elif key == self.video_pause_character:
                    self.pause_after_current_frame = False
                    break
                elif key == self.video_skip_to_frame_character:
                    frame_ind: str = enterbox("Enter desired frame (1-" + str(self.frame_count))
                    if frame_ind is not None and int(frame_ind) >= 1 and int(frame_ind) <= self.frame_count:
                        self.skip_to_frame(int(frame_ind))
                    break

        # FPS timer
        current_time: timer = timer()
        target_period_between_frames: float = 1e3 * 1 / self.frames_per_second / self.video_speed_factor
        if self.fps_timer is None:
            fps_pause_time_in_milliseconds: int = int(target_period_between_frames)
        else:
            elapsed_time_in_milliseconds: int = 1e3 * (current_time - self.fps_timer)
            fps_pause_time_in_milliseconds: int = int(target_period_between_frames - elapsed_time_in_milliseconds) \
                if (target_period_between_frames - 1) > elapsed_time_in_milliseconds else 1

        self.fps_timer = timer()
        key = waitKey(fps_pause_time_in_milliseconds) & 0xFF
        if key == self.video_quit_character:
            self.stop_capture()
        elif key == self.video_pause_character:
            # pause video
            pause_after_current_frame = False
            new_key = waitKey(0) & 0xFF
            if new_key == self.video_pause_character:
                # unpause
                pass
            elif new_key == self.video_next_frame_character \
                    or new_key == self.video_previous_frame_character \
                    or new_key == self.video_skip_to_frame_character:
                self.pause_after_current_frame = True
        elif key == self.video_increase_speed_character:
            self.increase_video_speed()
            self.frame_number_overlay.update_text_buffer_line(line_number=1,
                                                              line='speed factor = ' + str(self.video_speed_factor))
        elif key == self.video_decrease_speed_character:
            self.decrease_video_speed()
            self.frame_number_overlay.update_text_buffer_line(line_number=1,
                                                              line='speed factor = ' + str(self.video_speed_factor))

    def skip_to_frame(self, target_frame: int):
        if self.video_capture.get(CAP_PROP_FRAME_COUNT) == 0:
            # cannot control video directly
            print('cannot control video stream')
            return
        self.video_capture.set(CAP_PROP_POS_FRAMES, target_frame - 1)
        self.update_current_frame_number()

    def overlay_frame_index_window(self, frame_information: FrameInformation):
        frame_nb_string: str = 'Frame ' + str(self.current_frame_number) + ' / ' + str(self.frame_count)
        self.frame_number_overlay.update_text_buffer_line(line_number=0,
                                                          line=frame_nb_string)
        self.frame_number_overlay.write_to_frame(frame_information.frame)
        return frame_information

    def get_frame(self) -> Optional[np.ndarray]:
        res, image = self.video_capture.read()
        if not res:
            return None
        return image

    def run(self):
        for frame_information in self:
            try:
                self.on_run(frame_information=frame_information)
            except StopIteration:
                return

    def on_run(self, frame_information: FrameInformation):
        if frame_information is not None:
            imshow(winname="webcam", mat=frame_information.frame)

    def runnable(self):
        self.on_run(next(self))

    def update_current_frame_number(self):
        if self.frame_counter is None:
            self.current_frame_number = int(self.video_capture.get(CAP_PROP_POS_FRAMES))
        else:
            self.current_frame_number = next(self.frame_counter)

    def get_frame_count(self):
        if self.frame_counter is None:
            self.frame_count = int(self.video_capture.get(CAP_PROP_FRAME_COUNT))
        else:
            self.frame_count = -1

    def stop_capture(self):
        self.video_capture.release()
        destroyAllWindows()
        raise StopIteration

    def increase_video_speed(self):
        self.video_speed_factor *= 2

    def decrease_video_speed(self):
        self.video_speed_factor /= 2

    def get_frames_per_second(self) -> int:
        return self.frames_per_second

    def get_frame_width(self) -> int:
        return self.frame_width

    def get_frame_height(self) -> int:
        return self.frame_height + (self.BOTTOM_TEXT_OVERLAY_AREA_HEIGHT_PX if self.add_frame_overlay_area else 0)
