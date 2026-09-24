#!/usr/bin/env python3
"""Live ZED left-camera MegaPose inference using OpenCV only."""

import csv
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import perf_counter, time_ns
import cv2
import numpy as np
from configs.config import COLOR_RANGES, FPS, OUT_RES
import torch
from contour_runner import ContourRunner
from mpose_runner import MegaPoseRunner


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cam_source", default="0",
                        help="ZED device index or path, e.g. 0 or /dev/video2")
    parser.add_argument("--mesh", required=True,
                        help="Mesh path or filename inside models/")
    parser.add_argument("--cam_file", default="zed_1080p_raw_calib.json",
                        help="Raw LEFT camera calibration path or filename in configs/")
    parser.add_argument("--model", default="megapose-1.0-RGB")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--megapose-batch-size", type=int, default=16)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.fps <= 0 or args.megapose_batch_size <= 0:
        parser.error("FPS and MegaPose batch size must be positive.")
    if not 0 <= args.iou_threshold <= 1:
        parser.error("IoU threshold must be between 0 and 1.")
    return args


def resolve_input(value, folder):
    path = Path(value).expanduser()
    for candidate in (path, ROOT / path, ROOT / folder / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find {value!r} in the current directory or {folder}/")


def calculate_iou(a, b):
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (4,) or b.shape != (4,):
        return 0.0
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    intersection = np.maximum(0.0, np.minimum(a[2:], b[2:]) -
                              np.maximum(a[:2], b[:2]))
    overlap = float(np.prod(intersection))
    area_a = float(np.prod(np.maximum(0.0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0.0, b[2:] - b[:2])))
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def extract_left(frame, size):
    width, height = size
    expected = (height, width * 2, 3)
    if frame is None or frame.shape != expected or frame.dtype != np.uint8:
        actual = None if frame is None else frame.shape
        raise RuntimeError(
            f"Expected ZED side-by-side BGR frame {expected}; got {actual}. "
            "Check the device and supported capture mode."
        )
    return frame[:, :width].copy()


def process_frame(mpose, detector, left_bgr, iou_threshold):
    rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    _, boxes = detector.estimate(rgb)
    if boxes is None or len(boxes) == 0:
        mpose.reset_tracking()
        return left_bgr.copy(), None, "NO_TARGET"

    bbox = boxes[0]
    if mpose.tracking_active:
        try:
            predicted = mpose.mpose_bboxes()
        except ValueError:
            predicted = []
        if calculate_iou(bbox, predicted) < iou_threshold:
            mpose.reset_tracking()

    if not mpose.tracking_active:
        mpose.load_detection(bbox)

    pose = mpose.estimate(rgb)
    if pose is None:
        mpose.reset_tracking()
        display = left_bgr.copy()
        status = "NO_POSE"
    else:
        display = mpose.draw_triaxis()  # Runner returns BGR.
        status = "TRACKING"

    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return display, pose, status


def main():
    args = parse_args()
    mesh_path = resolve_input(args.mesh, "models")
    camera_path = resolve_input(args.cam_file, "configs")
    calibration = json.loads(camera_path.read_text())
    size = tuple(calibration["img_size"])
    if size != tuple(OUT_RES):
        raise ValueError(
            f"Calibration size {size} must match configs.config.OUT_RES={OUT_RES}."
        )
    width, height = size

    mpose = MegaPoseRunner(
        mesh_path, "fiducial", args.model, camera_path,
        batch_size=args.megapose_batch_size,
    )
    detector = ContourRunner(COLOR_RANGES)
    source = int(args.cam_source) if args.cam_source.isdecimal() else args.cam_source
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    video = None
    pose_file = None
    window_created = False
    frame_id = 0
    started = perf_counter()

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open ZED camera: {args.cam_source}")

        # ZED UVC output contains both eyes side by side.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; backend-dependent.

        ok, frame = cap.read()
        read_time_ns = time_ns()
        if not ok:
            raise RuntimeError("Opened camera, but could not read its first frame.")
        extract_left(frame, size)  # Check actual dimensions, not just cap.get().

        output_dir = ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "zed_pose_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        video_path = output_dir / f"{stem}.mp4"
        pose_path = output_dir / f"{stem}.csv"
        video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps), size,
        )
        if not video.isOpened():
            raise RuntimeError(f"Cannot create video: {video_path}")

        pose_file = pose_path.open("w", newline="", encoding="utf-8")
        rows = csv.writer(pose_file)
        rows.writerow([
            "timestamp", "host_read_time_ns", "status",
            "x_camera_m", "y_camera_m", "z_camera_m",
            "roll_camera_deg", "pitch_camera_deg", "yaw_camera_deg",
            "inference_seconds",
        ])

        cv2.namedWindow("ZED MegaPose", cv2.WINDOW_NORMAL)
        window_created = True
        cv2.resizeWindow("ZED MegaPose", 1280, 720)
        print(f"Capture: {frame.shape[1]}x{frame.shape[0]}, "
              f"reported FPS: {cap.get(cv2.CAP_PROP_FPS):g}")
        print(f"Left image: {width}x{height}; calibration: {camera_path}")
        print(f"Video: {video_path}\nPoses: {pose_path}")
        print("Press q or Escape to quit; r to reset tracking.")
        print(f"MP4 playback uses fixed {args.fps} FPS; CSV stores host read times.")
        previous_frame_time = perf_counter()

        while True:
            left_bgr = extract_left(frame, size)
            inference_start = perf_counter()
            try:
                display, pose, status = process_frame(
                    mpose, detector, left_bgr, args.iou_threshold
                )
            except torch.cuda.OutOfMemoryError:
                mpose.reset_tracking()
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory. Retry with --megapose-batch-size 4."
                ) from None
            inference_seconds = perf_counter() - inference_start

            now = perf_counter()
            loop_fps = 1.0 / max(now - previous_frame_time, 1e-6)
            previous_frame_time = now
            lines = [f"{status} | processed FPS: {loop_fps:.1f}"]
            if pose is not None:
                lines.append(
                    f"x={pose[0]:.3f} y={pose[1]:.3f} z={pose[2]:.3f} m"
                )
            for index, line in enumerate(lines):
                cv2.putText(display, line, (20, 35 + index * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            values = pose if pose is not None else [float("nan")] * 6
            rows.writerow([
                frame_id, read_time_ns, status, *values,
                f"{inference_seconds:.6f}",
            ])
            pose_file.flush()
            video.write(display)
            cv2.imshow("ZED MegaPose", display)
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                mpose.reset_tracking()
            if cv2.getWindowProperty("ZED MegaPose", cv2.WND_PROP_VISIBLE) < 1:
                break

            ok, frame = cap.read()
            read_time_ns = time_ns()  # Host arrival time, not sensor exposure time.
            if not ok:
                raise RuntimeError("ZED capture stopped returning frames.")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if video is not None:
            video.release()
        if pose_file is not None:
            pose_file.close()
        if window_created:
            cv2.destroyAllWindows()
        print(f"Processed {frame_id} frames in {perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    main()