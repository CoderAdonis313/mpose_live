import json
import numpy as np


STATUSES = {
    "STARTING",
    "TRACKING",
    "NO_TARGET",
    "NO_POSE",
    "RESET",
    "ERROR",
    "STOPPED",
}


class PacketGate:
    """Validate packets before updating ordering state."""

    def __init__(self, frame, object_id):
        self.frame = frame
        self.object_id = object_id
        self.session = None
        self.seq = -1
        self.result_ns = 0
        self.image_ns = 0


    def checked_transform(self, value):
        matrix = np.asarray(value, dtype=float)

        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Expected a finite 4x4 transform")

        rot = matrix[:3, :3]
        if (
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(rot), 1.0, atol=1e-3)
        ):
            raise ValueError("Invalid rigid transform")
        return matrix
    

    def accept(self, data, now_ns, packet_age_ns):
        p = json.loads(data)

        if not isinstance(p, dict) or type(p.get("schema_version")) is not int:
            raise ValueError("Missing packet schema")

        if p["schema_version"] != 1:
            raise ValueError("Unsupported packet schema")

        if (
            p.get("frame_id") != self.frame
            or p.get("object_id") != self.object_id
            or p.get("timestamp_source") != "host_read"
        ):
            raise ValueError("Unexpected frame, object, or timestamp source")

        session = p.get("session_id")
        seq = p.get("packet_seq")
        result_ns = p.get("result_time_ns")
        state = p.get("status")

        if not isinstance(session, str) or not session:
            raise ValueError("Missing session_id")

        if type(seq) is not int or seq < 0 or type(result_ns) is not int:
            raise ValueError("Invalid sequence or result timestamp")

        if not isinstance(state, str) or state not in STATUSES:
            raise ValueError("Unknown sender status")

        if type(p.get("valid")) is not bool or p["valid"] != (state == "TRACKING"):
            raise ValueError("Inconsistent validity flag")

        if (
            result_ns <= 0
            or result_ns > now_ns + 50_000_000
            or now_ns - result_ns > packet_age_ns
        ):
            raise ValueError("Old packet or incompatible clock")

        same_session = session == self.session
        if (
            (same_session and seq <= self.seq)
            or result_ns < self.result_ns
            or (not same_session and result_ns <= self.result_ns)
        ):
            raise ValueError("Duplicate or out-of-order packet")

        matrix = None

        if state in {"TRACKING", "NO_TARGET", "NO_POSE"}:
            image_ns = p.get("image_time_ns")
            index = p.get("frame_index")

            if (
                type(image_ns) is not int
                or image_ns <= 0
                or image_ns > result_ns
                or type(index) is not int
                or index < 0
            ):
                raise ValueError("Invalid source frame timestamp or index")

            if same_session and image_ns <= self.image_ns:
                raise ValueError("Non-increasing image timestamp")

        elif (
            p.get("image_time_ns") is not None
            or p.get("frame_index") is not None
        ):
            raise ValueError("Non-frame status must not contain an image timestamp")

        if p["valid"]:
            matrix = self.checked_transform(p.get("T_camera_object"))
            if matrix[2, 3] <= 0:
                raise ValueError("Object is behind camera")
        elif p.get("T_camera_object") is not None:
            raise ValueError("Invalid result contains a pose")

        self.session = session
        self.seq = seq
        self.result_ns = result_ns

        if not same_session:
            self.image_ns = 0

        if state in {"TRACKING", "NO_TARGET", "NO_POSE"}:
            self.image_ns = p["image_time_ns"]

        return p, matrix

