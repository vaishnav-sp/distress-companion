import unittest
from unittest.mock import Mock, patch

import numpy as np

from webcam import CameraCaptureSession, capture_frame, capture_frame_sequence


class WebcamCaptureTests(unittest.TestCase):
    @patch("webcam.cv2.VideoCapture")
    def test_reuses_one_open_camera_for_multiple_reads_and_closes_on_stop(
        self, video_capture: Mock
    ) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        camera = video_capture.return_value
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)

        session = CameraCaptureSession(camera_index=3)
        first = session.read()
        second = session.read()

        self.assertTrue(first.available)
        self.assertTrue(second.available)
        self.assertIs(first.frame, frame)
        self.assertIs(second.frame, frame)
        video_capture.assert_called_once_with(3)
        self.assertEqual(camera.read.call_count, 2)
        session.close()
        camera.release.assert_called_once()

    @patch("webcam.cv2.VideoCapture")
    def test_returns_frame_and_releases_camera(self, video_capture: Mock) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        camera = video_capture.return_value
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)

        result = capture_frame(camera_index=2)

        self.assertTrue(result.available)
        self.assertIs(result.frame, frame)
        self.assertTrue(result.camera_opened)
        video_capture.assert_called_once_with(2)
        camera.release.assert_called_once()

    @patch("webcam.cv2.VideoCapture")
    def test_reports_unavailable_camera_and_releases_it(self, video_capture: Mock) -> None:
        camera = video_capture.return_value
        camera.isOpened.return_value = False

        result = capture_frame()

        self.assertFalse(result.available)
        self.assertIsNone(result.frame)
        self.assertFalse(result.camera_opened)
        self.assertIn("Could not open", result.error or "")
        camera.release.assert_called_once()

    @patch("webcam.cv2.VideoCapture")
    def test_reports_failed_frame_read(self, video_capture: Mock) -> None:
        camera = video_capture.return_value
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)

        result = capture_frame()

        self.assertFalse(result.available)
        self.assertIn("no frame", result.error or "")
        self.assertTrue(result.camera_opened)
        camera.release.assert_called_once()

    @patch("webcam.cv2.VideoCapture", side_effect=RuntimeError("camera driver failed"))
    def test_retains_camera_capture_exception_type(self, video_capture: Mock) -> None:
        result = capture_frame()

        self.assertFalse(result.available)
        self.assertIn("camera driver failed", result.error or "")
        self.assertEqual(result.error_type, "RuntimeError")

    @patch("webcam.time.sleep")
    @patch("webcam.cv2.VideoCapture")
    def test_captures_short_timestamped_sequence_and_releases_camera(
        self,
        video_capture: Mock,
        sleep: Mock,
    ) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        camera = video_capture.return_value
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame), (True, frame), (True, frame)]

        result = capture_frame_sequence(frame_count=3, interval_seconds=0.1)

        self.assertTrue(result.available)
        self.assertEqual(len(result.frames), 3)
        self.assertTrue(all(item[0] is frame for item in result.frames))
        self.assertTrue(
            all(
                earlier[1] <= later[1]
                for earlier, later in zip(result.frames, result.frames[1:])
            )
        )
        self.assertEqual(sleep.call_count, 2)
        camera.release.assert_called_once()

    @patch("webcam.cv2.VideoCapture")
    def test_sequence_reports_unavailable_camera(self, video_capture: Mock) -> None:
        camera = video_capture.return_value
        camera.isOpened.return_value = False

        result = capture_frame_sequence(frame_count=2)

        self.assertFalse(result.available)
        self.assertEqual(result.frames, ())
        self.assertIn("Could not open", result.error or "")
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
