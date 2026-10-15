import unittest
from threading import Event
from unittest.mock import patch

import numpy as np

from microphone import (
    MicrophoneCaptureStream,
    capture_audio_window,
    check_microphone,
    test_microphone as record_microphone_test,
)


class MicrophoneTests(unittest.TestCase):
    @patch("microphone.sd.InputStream")
    def test_stream_reuses_device_and_yields_fresh_bounded_windows(
        self, input_stream: object
    ) -> None:
        stream = input_stream.return_value
        capture = MicrophoneCaptureStream(duration_seconds=0.00025)
        callback = input_stream.call_args.kwargs["callback"]
        callback(np.ones((4, 1), dtype=np.float32) * 0.25, 4, None, None)
        first = capture.read_window(timeout_seconds=0.001)
        callback(np.ones((4, 1), dtype=np.float32) * 0.5, 4, None, None)
        second = capture.read_window(timeout_seconds=0.001)

        self.assertTrue(first.available)
        self.assertTrue(second.available)
        self.assertAlmostEqual(first.rms_level or 0.0, 0.25)
        self.assertAlmostEqual(second.rms_level or 0.0, 0.5)
        self.assertEqual(input_stream.call_count, 1)
        capture.close()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    @patch("microphone.sd.InputStream")
    def test_stream_wait_is_interruptible_when_monitoring_stops(
        self, input_stream: object
    ) -> None:
        capture = MicrophoneCaptureStream(duration_seconds=0.5)
        stop_event = Event()
        stop_event.set()

        result = capture.read_window(stop_event, timeout_seconds=0.001)

        self.assertFalse(result.available)
        self.assertIn("stopped", result.message.lower())
        capture.close()

    @patch("microphone.sd.query_devices", return_value={"max_input_channels": 1})
    def test_detects_available_input(self, query_devices: object) -> None:
        result = check_microphone()

        self.assertTrue(result.available)
        self.assertIn("available", result.message)

    @patch("microphone.sd.query_devices", return_value={"max_input_channels": 0})
    def test_handles_missing_input_without_recording(self, query_devices: object) -> None:
        with patch("microphone.sd.rec") as record:
            result = record_microphone_test()

        self.assertFalse(result.available)
        self.assertIn("No microphone", result.message)
        record.assert_not_called()

    @patch("microphone.sd.wait")
    @patch("microphone.sd.rec")
    @patch("microphone.sd.query_devices", return_value={"max_input_channels": 1})
    def test_records_only_a_short_in_memory_sample(
        self,
        query_devices: object,
        record: object,
        wait: object,
    ) -> None:
        signal = np.array([[0.0], [0.5], [-0.5], [0.0]], dtype=np.float32)
        record.return_value = signal

        result = record_microphone_test(duration_seconds=0.25)

        self.assertTrue(result.available)
        self.assertAlmostEqual(result.rms_level or 0.0, float(np.sqrt(0.125)))
        self.assertAlmostEqual(result.peak_level or 0.0, 0.5)
        record.assert_called_once_with(
            4_000,
            samplerate=16_000,
            channels=1,
            dtype="float32",
        )
        wait.assert_called_once()

    def test_rejects_nonpositive_recording_duration(self) -> None:
        with patch("microphone.sd.query_devices") as query_devices:
            result = record_microphone_test(duration_seconds=0)

        self.assertFalse(result.available)
        self.assertIn("greater than zero", result.message)
        query_devices.assert_not_called()

    @patch("microphone.sd.rec", side_effect=RuntimeError("recording denied"))
    @patch("microphone.sd.query_devices", return_value={"max_input_channels": 1})
    def test_recording_error_is_reported_as_unavailable(
        self,
        query_devices: object,
        record: object,
    ) -> None:
        result = capture_audio_window()

        self.assertFalse(result.available)
        self.assertIn("recording denied", result.message)
        self.assertEqual(result.error_type, "RuntimeError")

    @patch("microphone.sd.query_devices", side_effect=RuntimeError("device query failed"))
    def test_query_error_is_reported_as_unavailable(self, query_devices: object) -> None:
        result = check_microphone()

        self.assertFalse(result.available)
        self.assertIn("device query failed", result.message)
        self.assertEqual(result.error_type, "RuntimeError")


if __name__ == "__main__":
    unittest.main()
