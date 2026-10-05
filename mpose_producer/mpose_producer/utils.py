import json
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
import math


STATUSES = {
    "STARTING",
    "TRACKING",
    "NO_TARGET",
    "NO_POSE",
    "RESET",
    "ERROR",
    "STOPPED",
}


def checked_transform(value):
    matrix = np.asarray(
        value,
        dtype=float,
    )

    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 transform")

    rotation = matrix[:3, :3]

    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-5,
    ):
        raise ValueError("Transform has an invalid final row")

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-3,
    ):
        raise ValueError("Transform rotation is not orthonormal")

    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-3,
    ):
        raise ValueError("Transform rotation determinant is not one")

    if matrix[2, 3] <= 0:
        raise ValueError("Object pose is behind the camera")

    return matrix


def checked_bbox(value, field_name):
    if value is None:
        return None

    bbox = np.asarray(
        value,
        dtype=float,
    )

    if (
        bbox.shape != (4,)
        or not np.isfinite(bbox).all()
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
    ):
        raise ValueError(f"{field_name} must be a valid " "[x1, y1, x2, y2] bbox")

    return bbox


class MultiPosePacketGate:
    """Validate and order multi-marker MegaPose packets."""

    def __init__(self, expected_frame):
        self.expected_frame = expected_frame

        self.session = None
        self.sequence = -1
        self.send_time_ns = 0

        # Prevent a delayed packet from an old producer session
        # from becoming active again.
        self.retired_sessions = set()

    @staticmethod
    def _required_integer(
        packet,
        key,
        minimum=0,
    ):
        value = packet.get(key)

        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")

        return value

    def accept(
        self,
        raw_data,
        now_ns,
        maximum_packet_age_ns,
    ):
        try:
            packet = json.loads(raw_data)
        except (
            ValueError,
            UnicodeDecodeError,
        ) as error:
            raise ValueError("Packet is not valid UTF-8 JSON") from error

        if not isinstance(packet, dict):
            raise ValueError("UDP packet must contain a JSON object")

        if type(packet.get("schema_version")) is not int:
            raise ValueError("Missing packet schema version")

        if packet["schema_version"] != 1:
            raise ValueError("Unsupported packet schema version")

        if packet.get("timestamp_source") != "host_read":
            raise ValueError("Unsupported timestamp source")

        if packet.get("frame_id") != self.expected_frame:
            raise ValueError("Unexpected camera frame")

        session = packet.get("session_id")

        if not isinstance(session, str) or not session:
            raise ValueError("Missing session_id")

        if session in self.retired_sessions:
            raise ValueError("Packet belongs to a retired session")

        sequence = self._required_integer(
            packet,
            "packet_seq",
        )

        send_time_ns = self._required_integer(
            packet,
            "send_time_ns",
            minimum=1,
        )

        if send_time_ns > now_ns + 50_000_000:
            raise ValueError("Packet timestamp is in the future")

        if now_ns - send_time_ns > maximum_packet_age_ns:
            raise ValueError("UDP packet is too old")

        status = packet.get("status")
        source_status = packet.get("source_status")

        valid = packet.get("valid")
        complete = packet.get("complete")

        if type(valid) is not bool:
            raise ValueError("valid must be a boolean")

        if type(complete) is not bool:
            raise ValueError("complete must be a boolean")

        frame_index = packet.get("frame_index")
        image_time_ns = packet.get("image_time_ns")
        result_time_ns = packet.get("result_time_ns")

        frame_values = (
            frame_index,
            image_time_ns,
            result_time_ns,
        )

        has_frame_metadata = all(value is not None for value in frame_values)

        if any(value is not None for value in frame_values) and not has_frame_metadata:
            raise ValueError("Frame metadata is incomplete")

        if has_frame_metadata:
            if (
                type(frame_index) is not int
                or frame_index < 0
                or type(image_time_ns) is not int
                or image_time_ns <= 0
                or type(result_time_ns) is not int
                or result_time_ns <= 0
            ):
                raise ValueError("Invalid frame metadata")

            if not (image_time_ns <= result_time_ns <= send_time_ns):
                raise ValueError("Inconsistent frame timestamps")

        poses_value = packet.get("poses")

        if not isinstance(poses_value, dict):
            raise ValueError("poses must be a dictionary")

        pose_count = packet.get("pose_count")

        if type(pose_count) is not int or pose_count != len(poses_value):
            raise ValueError("pose_count does not match poses")

        if poses_value and not has_frame_metadata:
            raise ValueError("Pose data requires source frame metadata")

        expected_valid = status in {"TRACKING", "PARTIAL"} and bool(poses_value)

        if valid != expected_valid:
            raise ValueError(
                "Packet validity is inconsistent " "with its status and poses"
            )

        expected_complete = valid and source_status == "TRACKING"

        if complete != expected_complete:
            raise ValueError("Packet completeness flag is inconsistent")

        if poses_value and status not in {
            "TRACKING",
            "PARTIAL",
            "STALE",
        }:
            raise ValueError("Unexpected poses for this status")

        parsed_poses = {}

        for label, pose_value in poses_value.items():
            if not isinstance(pose_value, dict):
                raise ValueError(f"Pose for {label!r} must be " "a dictionary")

            marker_valid = pose_value.get("valid")

            if type(marker_valid) is not bool:
                raise ValueError(f"Pose validity for {label!r} " "must be boolean")

            if marker_valid != valid:
                raise ValueError(
                    f"Pose validity for {label!r} " "does not match packet validity"
                )

            matrix = checked_transform(pose_value.get("T_camera_object"))

            checked_bbox(
                pose_value.get("detection_bbox"),
                f"{label}.detection_bbox",
            )

            checked_bbox(
                pose_value.get("projected_bbox"),
                f"{label}.projected_bbox",
            )

            parsed_poses[label] = {
                "matrix": matrix,
                "valid": marker_valid,
            }

        same_session = session == self.session

        if same_session:
            if sequence <= self.sequence:
                raise ValueError("Duplicate or out-of-order packet")

            if send_time_ns < self.send_time_ns:
                raise ValueError("Packet send timestamp moved backwards")

        # Update ordering state only after complete validation.
        if not same_session:
            if self.session is not None:
                self.retired_sessions.add(self.session)

            self.session = session
            self.sequence = -1
            self.send_time_ns = 0

        self.sequence = sequence
        self.send_time_ns = send_time_ns

        return packet, parsed_poses
