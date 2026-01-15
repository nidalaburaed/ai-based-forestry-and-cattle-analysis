from typing import Iterator, List, Optional, Tuple, Union
import os
import time
import cv2
import argparse

class VideoReader:
    def __init__(self, video_path: Union[str, int], interval_seconds: int = 30, out_dir: Optional[str] = "frames", prefix: str = "frame", image_ext: str = "jpg", jpeg_quality: int = 95):
        # Convert string camera indices to integers (e.g., "0" -> 0)
        if isinstance(video_path, str) and video_path.isdigit():
            video_path = int(video_path)

        self.video_path = video_path
        self.interval_seconds = interval_seconds
        self.out_dir = out_dir
        self.prefix = prefix
        self.image_ext = image_ext
        self.jpeg_quality = jpeg_quality

        # Try to open video/camera with multiple backends for better compatibility
        self.cap = self._open_video_source(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video/stream: {video_path}")

    def _open_video_source(self, video_path: Union[str, int]) -> cv2.VideoCapture:
        """
        Try to open video source with multiple backends for better compatibility.

        For camera indices on Windows, tries multiple backends:
        - CAP_DSHOW (DirectShow) - usually most reliable on Windows
        - CAP_MSMF (Media Foundation) - default on Windows
        - Default backend
        """
        # If it's a camera index (integer), try different backends
        if isinstance(video_path, int):
            # Try DirectShow first (usually most reliable on Windows)
            cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()

            # Try Media Foundation
            cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
            if cap.isOpened():
                return cap
            cap.release()

            # Try default backend
            cap = cv2.VideoCapture(video_path)
            return cap
        else:
            # For file paths or URLs, use default backend
            return cv2.VideoCapture(video_path)

    def _sec_to_hms(self, sec: int) -> str:
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}-{m:02d}-{s:02d}"

    def read_frames(self, interval_seconds: Optional[int] = None) -> Iterator[Tuple[int, float, "cv2.Mat"]]:
        """
        Generator that yields frames at specified intervals.

        Args:
            interval_seconds: Seconds between frames. If None, uses self.interval_seconds

        Yields:
            Tuple of (frame_number, timestamp, frame)
        """
        interval = interval_seconds if interval_seconds is not None else self.interval_seconds

        if interval <= 0:
            raise ValueError("interval_seconds must be > 0")

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = None
        if fps > 0 and frame_count > 0:
            duration = int(frame_count / fps)

        # For video files (not streams)
        if duration is not None:
            t = 0
            frame_num = 0
            while True:
                if t > duration:
                    break
                self.cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield (frame_num, float(t), frame)
                t += interval
                frame_num += 1
        else:
            # For streams
            start_time = time.time()
            last_yield = start_time - interval
            frame_num = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                now = time.time()
                if now - last_yield >= interval:
                    timestamp = now - start_time
                    last_yield = now
                    yield (frame_num, timestamp, frame)
                    frame_num += 1

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()

    def extract_frames(self) -> List[str] or Iterator[Tuple[int, "cv2.Mat"]]:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = None
        if fps > 0 and frame_count > 0:
            duration = int(frame_count / fps)

        def _yield_frames_file():
            try:
                t = 0
                while True:
                    if duration is not None and t > duration:
                        break
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    yield (t, frame)
                    t += self.interval_seconds
            finally:
                self.cap.release()

        def _yield_frames_stream():
            start_time = time.time()
            last_yield = start_time - self.interval_seconds
            try:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                    now = time.time()
                    if now - last_yield >= self.interval_seconds:
                        timestamp = int(now - start_time)
                        last_yield = now
                        yield (timestamp, frame)
            finally:
                self.cap.release()

        if self.out_dir is None:
            if duration is None:
                return _yield_frames_stream()
            else:
                return _yield_frames_file()

        os.makedirs(self.out_dir, exist_ok=True)
        saved_paths: List[str] = []
        encode_params = []
        if self.image_ext.lower() in ("jpg", "jpeg"):
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]

        generator = _yield_frames_stream() if duration is None else _yield_frames_file()

        for t, frame in generator:
            fname = f"{self.prefix}_{self._sec_to_hms(int(t))}.{self.image_ext}"
            out_path = os.path.join(self.out_dir, fname)
            ok = cv2.imwrite(out_path, frame, encode_params) if encode_params else cv2.imwrite(out_path, frame)
            if not ok:
                continue
            saved_paths.append(out_path)

        return saved_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames every N seconds from a video or stream.")
    parser.add_argument("video", help="Path to video file, stream URL or camera index (0,1,...).")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between frames (default 30)")
    parser.add_argument("--out", default="frames", help="Output directory (use 'None' to yield frames instead of saving)")
    args = parser.parse_args()

    out_arg = None if isinstance(args.out, str) and args.out.lower() == "none" else args.out
    video_reader = VideoReader(args.video, interval_seconds=args.interval, out_dir=out_arg)
    result = video_reader.extract_frames()
    if out_arg is not None:
        print(f"Saved {len(result)} frames to {out_arg}")
    else:
        count = 0
        for t, _ in result:
            print(f"Yielded frame at {t}s")
            count += 1
        print(f"Yielded {count} frames")
