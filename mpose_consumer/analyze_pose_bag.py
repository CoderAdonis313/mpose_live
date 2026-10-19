#!/usr/bin/env python3
"""Analyze MegaPose accuracy and runtime telemetry from a ROS 2 bag.

Automatically pairs:
    /relative_pose/estimated/<bot_label>
    /relative_pose/ground_truth/<bot_label>

Reports average errors and detailed statistics separately for each bot.
Also supports legacy unsuffixed topics and explicitly selected topic pairs.

Supported inputs:
    SQLite .db3 file
    MCAP .mcap file
    Rosbag directory containing .db3 or .mcap files

Translation errors are in meters.
Geodesic rotation errors are in degrees.

Runtime telemetry is read from /mpose/packets when available.
"Frames missed" assumes the marker should be visible throughout the interval.
"""

import argparse
import bisect
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct


POSE_TYPE = "geometry_msgs/msg/PoseStamped"
STRING_TYPE = "std_msgs/msg/String"
SERIALIZATION_FORMAT = "cdr"


def cdr_order_and_payload(data):
    """Return byte order and CDR payload."""
    data = bytes(data)

    if len(data) < 4 or data[:2] not in (b"\x00\x01", b"\x00\x00"):
        raise ValueError("Expected standard CDR v1 encapsulation")

    order = "<" if data[1] == 1 else ">"
    return order, memoryview(data)[4:]


def decode_pose(data):
    """Decode a CDR geometry_msgs/msg/PoseStamped message."""
    order, payload = cdr_order_and_payload(data)

    if len(payload) < 12:
        raise ValueError("PoseStamped payload is too short")

    sec, nanosec, length = struct.unpack_from(order + "iII", payload, 0)

    if nanosec >= 1_000_000_000:
        raise ValueError("Invalid nanosecond value")

    if length < 1 or 12 + length > len(payload):
        raise ValueError("Invalid frame_id string length")

    frame_bytes = bytes(payload[12:12 + length])

    if frame_bytes[-1:] != b"\x00":
        raise ValueError("frame_id is not null terminated")

    frame = frame_bytes[:-1].decode("utf-8")
    offset = 12 + length
    offset += (-offset) % 8

    if offset + 56 > len(payload):
        raise ValueError("PoseStamped pose payload is too short")

    values = struct.unpack_from(order + "7d", payload, offset)
    header_ns = sec * 1_000_000_000 + nanosec

    return header_ns, frame, values[:3], values[3:]


def decode_string(data):
    """Decode a CDR std_msgs/msg/String message."""
    order, payload = cdr_order_and_payload(data)

    if len(payload) < 4:
        raise ValueError("String payload is too short")

    length = struct.unpack_from(order + "I", payload, 0)[0]

    if length < 1 or 4 + length > len(payload):
        raise ValueError("Invalid String length")

    value = bytes(payload[4:4 + length])

    if value[-1:] != b"\x00":
        raise ValueError("String is not null terminated")

    return value[:-1].decode("utf-8")


def normalize(quaternion):
    """Return a normalized quaternion."""
    norm = math.sqrt(sum(value * value for value in quaternion))

    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")

    return tuple(value / norm for value in quaternion)


def discover_bag(path):
    """Find bag storage files and determine their format."""
    if path.is_file():
        suffix = path.suffix.lower()

        if suffix == ".db3":
            return "sqlite3", [path]

        if suffix == ".mcap":
            return "mcap", [path]

        raise ValueError(
            "Supply a .db3 file, .mcap file, or rosbag directory"
        )

    if not path.is_dir():
        raise ValueError("Bag path does not exist")

    sqlite_paths = sorted(
        candidate
        for candidate in path.glob("*.db3")
        if candidate.is_file()
    )
    mcap_paths = sorted(
        candidate
        for candidate in path.glob("*.mcap")
        if candidate.is_file()
    )

    if sqlite_paths and mcap_paths:
        raise ValueError("Bag contains both SQLite and MCAP files")

    if sqlite_paths:
        return "sqlite3", sqlite_paths

    if mcap_paths:
        return "mcap", mcap_paths

    raise ValueError(
        "Bag directory contains no .db3 or .mcap files. "
        "Select an individual bag directory, not its parent."
    )


def read_sqlite_messages(paths, topic_name, expected_type, required=True):
    """Read serialized messages from SQLite bag parts."""
    messages = []
    found = False

    for path in paths:
        uri = path.resolve().as_uri() + "?mode=ro"

        with sqlite3.connect(uri, uri=True) as connection:
            topic = connection.execute(
                """
                SELECT id, type, serialization_format
                FROM topics
                WHERE name = ?
                """,
                (topic_name,),
            ).fetchone()

            if topic is None:
                continue

            found = True
            topic_id, topic_type, serialization_format = topic
            actual = topic_type, serialization_format
            expected = expected_type, SERIALIZATION_FORMAT

            if actual != expected:
                raise ValueError(
                    f"{topic_name}: expected {expected}, found {actual}"
                )

            rows = connection.execute(
                """
                SELECT timestamp, data
                FROM messages
                WHERE topic_id = ?
                ORDER BY timestamp
                """,
                (topic_id,),
            )
            messages.extend(rows)

    if required and not found:
        raise ValueError(f"Topic not found: {topic_name}")

    messages.sort(key=lambda item: item[0])
    return messages, found


def import_rosbag2_py():
    """Import rosbag2_py only when MCAP is used."""
    try:
        import rosbag2_py
    except ImportError as error:
        raise ValueError(
            "MCAP input requires rosbag2_py and the MCAP storage plugin. "
            "Source ROS before running."
        ) from error

    return rosbag2_py


def open_mcap_reader(path, rosbag2_py):
    """Open one MCAP file."""
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(path.resolve()),
        storage_id="mcap",
    )
    converter_options = rosbag2_py.ConverterOptions("", "")

    try:
        reader.open(storage_options, converter_options)
    except RuntimeError as error:
        raise ValueError(
            f"Cannot open MCAP file {path}: {error}"
        ) from error

    return reader


def read_mcap_messages(paths, topic_name, expected_type, required=True):
    """Read serialized messages from MCAP bag parts."""
    rosbag2_py = import_rosbag2_py()
    messages = []
    found = False

    for path in paths:
        reader = open_mcap_reader(path, rosbag2_py)
        topics = {
            topic.name: topic
            for topic in reader.get_all_topics_and_types()
        }
        topic = topics.get(topic_name)

        if topic is None:
            continue

        found = True
        actual = topic.type, topic.serialization_format
        expected = expected_type, SERIALIZATION_FORMAT

        if actual != expected:
            raise ValueError(
                f"{topic_name}: expected {expected}, found {actual}"
            )

        try:
            while reader.has_next():
                name, data, bag_ns = reader.read_next()

                if name == topic_name:
                    messages.append((bag_ns, data))
        except RuntimeError as error:
            raise ValueError(
                f"Error while reading {path}: {error}"
            ) from error

    if required and not found:
        raise ValueError(f"Topic not found: {topic_name}")

    messages.sort(key=lambda item: item[0])
    return messages, found


def read_serialized_messages(
    paths, storage, topic_name, expected_type, required=True
):
    """Read serialized messages from either bag format."""
    if storage == "sqlite3":
        return read_sqlite_messages(
            paths,
            topic_name,
            expected_type,
            required=required,
        )

    if storage == "mcap":
        return read_mcap_messages(
            paths,
            topic_name,
            expected_type,
            required=required,
        )

    raise ValueError(f"Unsupported storage format: {storage}")


def read_pose_topic(paths, storage, topic_name, clock):
    """Decode and validate one PoseStamped topic."""
    messages, _ = read_serialized_messages(
        paths,
        storage,
        topic_name,
        POSE_TYPE,
        required=True,
    )

    samples = []
    invalid = 0

    for bag_ns, data in messages:
        try:
            header_ns, frame, translation, quaternion = decode_pose(data)
            quaternion = normalize(quaternion)
        except (UnicodeDecodeError, ValueError, struct.error):
            invalid += 1
            continue

        stamp = header_ns if clock == "header" else bag_ns
        values = (*translation, *quaternion)

        if stamp <= 0 or not all(math.isfinite(value) for value in values):
            invalid += 1
            continue

        samples.append(
            (
                stamp,
                frame,
                translation,
                quaternion,
                bag_ns,
                header_ns,
            )
        )

    samples.sort(key=lambda sample: sample[0])
    duplicates = len(samples) - len({sample[0] for sample in samples})

    if duplicates:
        raise ValueError(
            f"{topic_name}: {duplicates} duplicate timestamps"
        )

    if not samples:
        raise ValueError(f"No valid samples: {topic_name}")

    return samples, invalid


def read_packet_topic(paths, storage, topic_name):
    """Read and parse raw MegaPose packet JSON."""
    messages, found = read_serialized_messages(
        paths,
        storage,
        topic_name,
        STRING_TYPE,
        required=False,
    )

    packets = []
    invalid = 0

    for bag_ns, data in messages:
        try:
            packet = json.loads(decode_string(data))

            if not isinstance(packet, dict):
                raise ValueError("Packet must be a JSON object")

            if packet.get("schema_version") != 1:
                raise ValueError("Unsupported packet schema")

            session_id = packet.get("session_id")
            packet_seq = packet.get("packet_seq")
            send_time_ns = packet.get("send_time_ns")

            if (
                not isinstance(session_id, str)
                or not session_id
                or type(packet_seq) is not int
                or packet_seq < 0
                or type(send_time_ns) is not int
                or send_time_ns <= 0
            ):
                raise ValueError("Invalid packet identity")

        except (UnicodeDecodeError, ValueError, struct.error):
            invalid += 1
            continue

        packets.append((bag_ns, packet))

    return packets, invalid, found


def slerp(first, second, fraction):
    """Shortest-path quaternion SLERP."""
    dot = sum(left * right for left, right in zip(first, second))

    if dot < 0:
        second = tuple(-value for value in second)
        dot = -dot

    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        interpolated = tuple(
            (1 - fraction) * left + fraction * right
            for left, right in zip(first, second)
        )
        return normalize(interpolated)

    angle = math.acos(dot)
    denominator = math.sin(angle)
    left_weight = math.sin((1 - fraction) * angle) / denominator
    right_weight = math.sin(fraction * angle) / denominator

    return normalize(
        tuple(
            left_weight * left + right_weight * right
            for left, right in zip(first, second)
        )
    )


def percentile(values, proportion):
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    low = math.floor(index)
    high = math.ceil(index)

    return (
        ordered[low]
        + (ordered[high] - ordered[low]) * (index - low)
    )


def describe(values):
    """Return summary statistics or None for no values."""
    values = [
        float(value)
        for value in values
        if math.isfinite(value)
    ]

    if not values:
        return None

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "rmse": math.sqrt(
            statistics.fmean(value * value for value in values)
        ),
        "minimum": min(values),
        "maximum": max(values),
    }


def coordinate_ranges(samples):
    """Return minimum and maximum X/Y/Z."""
    return [
        [
            min(sample[index] for sample in samples),
            max(sample[index] for sample in samples),
        ]
        for index in range(3)
    ]


def pose_rate_hz(timestamps_ns):
    """Calculate effective sample rate from unique timestamps."""
    timestamps_ns = sorted(set(timestamps_ns))

    if len(timestamps_ns) < 2:
        return None

    duration_seconds = (
        timestamps_ns[-1] - timestamps_ns[0]
    ) / 1e9

    if duration_seconds <= 0:
        return None

    return (len(timestamps_ns) - 1) / duration_seconds


def nonnegative_integer(value):
    """Return a valid counter value or None."""
    if type(value) is int and value >= 0:
        return value

    return None


def update_maximum(mapping, key, value):
    """Update a cumulative counter maximum."""
    value = nonnegative_integer(value)

    if value is None:
        return

    mapping[key] = max(mapping.get(key, 0), value)


def summarize_telemetry(packets):
    """Combine cumulative telemetry across producer sessions."""
    scalar_fields = (
        "submitted_inputs",
        "processed_inputs",
        "dropped_inputs",
        "manual_resets",
        "camera_frames_skipped",
    )

    marker_fields = (
        "automatic_resets",
        "processed_without_pose",
        "detected_without_pose",
        "dropped_detected_inputs",
    )

    sessions = {}

    for _, packet in packets:
        telemetry = packet.get("telemetry")

        if not isinstance(telemetry, dict):
            continue

        session_id = packet["session_id"]
        session = sessions.setdefault(
            session_id,
            {
                "scalars": {},
                "markers": {
                    field: {} for field in marker_fields
                },
            },
        )

        for field in scalar_fields:
            update_maximum(
                session["scalars"],
                field,
                telemetry.get(field),
            )

        for field in marker_fields:
            values = telemetry.get(field)

            if not isinstance(values, dict):
                continue

            for label, value in values.items():
                if not isinstance(label, str):
                    continue

                update_maximum(
                    session["markers"][field],
                    label,
                    value,
                )

    if not sessions:
        return {
            "available": False,
            "session_count": 0,
        }

    totals = {field: 0 for field in scalar_fields}
    marker_totals = {field: {} for field in marker_fields}

    for session in sessions.values():
        for field in scalar_fields:
            totals[field] += session["scalars"].get(field, 0)

        for field in marker_fields:
            for label, value in session["markers"][field].items():
                marker_totals[field][label] = (
                    marker_totals[field].get(label, 0) + value
                )

    submitted = totals["submitted_inputs"]
    processed = totals["processed_inputs"]
    camera_skipped = totals["camera_frames_skipped"]
    eligible_camera_frames = camera_skipped + submitted

    labels = sorted(
        set().union(
            *(set(values) for values in marker_totals.values())
        )
    )

    per_marker = {}

    for label in labels:
        processed_without_pose = marker_totals[
            "processed_without_pose"
        ].get(label, 0)

        successful_pose_frames = max(
            0, processed - processed_without_pose
        )

        missed_frames = max(
            0, eligible_camera_frames - successful_pose_frames
        )

        missed_percentage = (
            100.0 * missed_frames / eligible_camera_frames
            if eligible_camera_frames > 0
            else None
        )

        automatic_resets = marker_totals[
            "automatic_resets"
        ].get(label, 0)

        per_marker[label] = {
            "successful_pose_frames": successful_pose_frames,
            "frames_missed_assuming_visible": missed_frames,
            "frames_missed_percentage_assuming_visible": missed_percentage,
            "automatic_resets": automatic_resets,
            "manual_global_resets": totals["manual_resets"],
            "total_resets_affecting_marker": (
                automatic_resets + totals["manual_resets"]
            ),
            "processed_without_pose": processed_without_pose,
            "detected_without_pose": marker_totals[
                "detected_without_pose"
            ].get(label, 0),
            "dropped_detected_inputs": marker_totals[
                "dropped_detected_inputs"
            ].get(label, 0),
        }

    return {
        "available": True,
        "session_count": len(sessions),
        **totals,
        "eligible_camera_frames_assuming_visible": eligible_camera_frames,
        "per_marker": per_marker,
    }


def summarize_packet_timing(packets):
    """Summarize timing from unique inference results."""
    unique_results = {}

    for bag_ns, packet in packets:
        session_id = packet.get("session_id")
        frame_index = packet.get("frame_index")
        image_time_ns = packet.get("image_time_ns")
        result_time_ns = packet.get("result_time_ns")

        if (
            type(frame_index) is not int
            or type(image_time_ns) is not int
            or type(result_time_ns) is not int
            or image_time_ns <= 0
            or result_time_ns < image_time_ns
        ):
            continue

        key = (
            session_id,
            frame_index,
            image_time_ns,
            result_time_ns,
        )
        previous = unique_results.get(key)

        # Prefer a valid publication over a stale heartbeat.
        if previous is None or (
            packet.get("valid") is True
            and previous[1].get("valid") is not True
        ):
            unique_results[key] = bag_ns, packet

    inference_ms = []
    image_to_result_ms = []
    image_to_send_ms = []
    send_to_bag_ms = []
    marker_timestamps = {}

    for bag_ns, packet in unique_results.values():
        inference_seconds = packet.get("inference_seconds")

        if (
            isinstance(inference_seconds, (int, float))
            and math.isfinite(inference_seconds)
            and inference_seconds >= 0
        ):
            inference_ms.append(float(inference_seconds) * 1000.0)

        image_time_ns = packet["image_time_ns"]
        result_time_ns = packet["result_time_ns"]
        send_time_ns = packet.get("send_time_ns")

        image_to_result_ms.append(
            (result_time_ns - image_time_ns) / 1e6
        )

        if (
            type(send_time_ns) is int
            and send_time_ns >= result_time_ns
        ):
            image_to_send_ms.append(
                (send_time_ns - image_time_ns) / 1e6
            )
            send_to_bag_ms.append(
                (bag_ns - send_time_ns) / 1e6
            )

        if packet.get("valid") is not True:
            continue

        poses = packet.get("poses")

        if not isinstance(poses, dict):
            continue

        for label, pose in poses.items():
            if (
                isinstance(label, str)
                and isinstance(pose, dict)
                and pose.get("valid") is True
            ):
                marker_timestamps.setdefault(label, []).append(
                    image_time_ns
                )

    return {
        "unique_inference_results": len(unique_results),
        "inference_ms": describe(inference_ms),
        "image_to_result_ms": describe(image_to_result_ms),
        "image_to_send_ms": describe(image_to_send_ms),
        "send_to_bag_receipt_ms": describe(send_to_bag_ms),
        "effective_pose_rate_hz_by_marker": {
            label: pose_rate_hz(timestamps)
            for label, timestamps in sorted(marker_timestamps.items())
        },
        "valid_unique_pose_results_by_marker": {
            label: len(set(timestamps))
            for label, timestamps in sorted(marker_timestamps.items())
        },
    }


def match_pose_samples(
    estimated,
    ground_truth,
    method,
    max_gap_ms,
    offset_ms,
    beta,
    translation_threshold_m,
    rotation_threshold_deg,
):
    """Match estimates to ground truth and calculate errors."""
    ground_truth_stamps = [
        sample[0] for sample in ground_truth
    ]
    max_gap_ns = round(max_gap_ms * 1e6)
    offset_ns = round(offset_ms * 1e6)

    records = []
    used_ground_truth = []
    skipped = {
        "outside_ground_truth_range": 0,
        "gap_too_large": 0,
    }

    for sample in estimated:
        stamp, _, estimated_t, estimated_q, _, _ = sample
        matching_stamp = stamp + offset_ns

        if (
            matching_stamp < ground_truth_stamps[0]
            or matching_stamp > ground_truth_stamps[-1]
        ):
            skipped["outside_ground_truth_range"] += 1
            continue

        right = bisect.bisect_left(
            ground_truth_stamps, matching_stamp
        )

        if ground_truth_stamps[right] == matching_stamp:
            ground_truth_t = ground_truth[right][2]
            ground_truth_q = ground_truth[right][3]
            gap_ns = 0

        elif method == "nearest":
            chosen = min(
                (right - 1, right),
                key=lambda index: abs(
                    ground_truth_stamps[index] - matching_stamp
                ),
            )

            gap_ns = abs(
                ground_truth_stamps[chosen] - matching_stamp
            )
            ground_truth_t = ground_truth[chosen][2]
            ground_truth_q = ground_truth[chosen][3]

        else:
            left = right - 1
            left_stamp = ground_truth_stamps[left]
            right_stamp = ground_truth_stamps[right]

            gap_ns = max(
                matching_stamp - left_stamp,
                right_stamp - matching_stamp,
            )
            fraction = (
                (matching_stamp - left_stamp)
                / (right_stamp - left_stamp)
            )

            ground_truth_t = tuple(
                (1 - fraction) * left_value
                + fraction * right_value
                for left_value, right_value in zip(
                    ground_truth[left][2],
                    ground_truth[right][2],
                )
            )
            ground_truth_q = slerp(
                ground_truth[left][3],
                ground_truth[right][3],
                fraction,
            )

        if gap_ns > max_gap_ns:
            skipped["gap_too_large"] += 1
            continue

        delta = tuple(
            estimated_value - ground_truth_value
            for estimated_value, ground_truth_value in zip(
                estimated_t, ground_truth_t
            )
        )
        translation_squared = sum(
            value * value for value in delta
        )

        quaternion_dot = abs(
            sum(
                estimated_value * ground_truth_value
                for estimated_value, ground_truth_value in zip(
                    estimated_q, ground_truth_q
                )
            )
        )
        quaternion_dot = min(1.0, max(0.0, quaternion_dot))

        # q and -q represent the same physical orientation.
        quaternion_squared = min(
            sum(
                (estimated_value - ground_truth_value) ** 2
                for estimated_value, ground_truth_value in zip(
                    estimated_q, ground_truth_q
                )
            ),
            sum(
                (estimated_value + ground_truth_value) ** 2
                for estimated_value, ground_truth_value in zip(
                    estimated_q, ground_truth_q
                )
            ),
        )

        translation_error = math.sqrt(translation_squared)
        quaternion_error = math.sqrt(quaternion_squared)
        rotation_error = math.degrees(
            2.0 * math.acos(quaternion_dot)
        )

        records.append(
            {
                "timestamp_ns": stamp,
                "matching_timestamp_ns": matching_stamp,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "squared_translation_error_m2": translation_squared,
                "squared_quaternion_error": quaternion_squared,
                "pose_loss": (
                    translation_error + beta * quaternion_error
                ),
                "ground_truth_sample_distance_ms": gap_ns / 1e6,
                "delta_xyz_m": delta,
                "within_accuracy_thresholds": (
                    translation_error <= translation_threshold_m
                    and rotation_error <= rotation_threshold_deg
                ),
            }
        )
        used_ground_truth.append(ground_truth_t)

    return records, used_ground_truth, skipped


def analyze_topic_pair(args):
    """Analyze one estimated and ground-truth topic pair."""
    storage, paths = discover_bag(args.bag)

    estimated, invalid_estimated = read_pose_topic(
        paths,
        storage,
        args.estimated_topic,
        args.clock,
    )
    ground_truth, invalid_ground_truth = read_pose_topic(
        paths,
        storage,
        args.ground_truth_topic,
        args.clock,
    )

    estimated_frames = {sample[1] for sample in estimated}
    ground_truth_frames = {sample[1] for sample in ground_truth}

    if (
        len(estimated_frames) != 1
        or estimated_frames != ground_truth_frames
        or "" in estimated_frames
    ):
        raise ValueError(
            "Expected one identical nonempty frame_id: "
            f"estimated={estimated_frames}, "
            f"ground_truth={ground_truth_frames}"
        )

    records, used_ground_truth, skipped = match_pose_samples(
        estimated=estimated,
        ground_truth=ground_truth,
        method=args.method,
        max_gap_ms=args.max_gap_ms,
        offset_ms=args.estimate_time_offset_ms,
        beta=args.beta,
        translation_threshold_m=args.translation_threshold_m,
        rotation_threshold_deg=args.rotation_threshold_deg,
    )

    if not records:
        raise ValueError(
            f"No matched samples. Skipped: {skipped}"
        )

    packets, invalid_packets, packet_found = read_packet_topic(
        paths,
        storage,
        args.packet_topic,
    )
    telemetry = summarize_telemetry(packets)
    timing = summarize_packet_timing(packets)

    selected_marker_metrics = None

    if args.marker_label:
        marker_metrics = telemetry.get("per_marker", {})

        if args.marker_label not in marker_metrics:
            raise ValueError(
                f"Marker {args.marker_label!r} is not "
                "present in packet telemetry"
            )

        selected_marker_metrics = marker_metrics[args.marker_label]

    translation_errors = [
        record["translation_error_m"] for record in records
    ]
    rotation_errors = [
        record["rotation_error_deg"] for record in records
    ]
    threshold_successes = sum(
        record["within_accuracy_thresholds"]
        for record in records
    )
    estimated_span_seconds = (
        estimated[-1][0] - estimated[0][0]
    ) / 1e9

    result = {
        "storage_identifier": storage,
        "bag_files": [str(path) for path in paths],
        "estimated_topic": args.estimated_topic,
        "ground_truth_topic": args.ground_truth_topic,
        "packet_topic": args.packet_topic,
        "packet_topic_found": packet_found,
        "frame_id": next(iter(estimated_frames)),
        "timestamp_source": args.clock,
        "matching_method": args.method,
        "maximum_sample_distance_ms": args.max_gap_ms,
        "estimate_time_offset_ms": args.estimate_time_offset_ms,
        "beta": args.beta,
        "pose_loss_definition": (
            "||t_est-t_gt|| + beta * "
            "min(||q_est-q_gt||, ||q_est+q_gt||)"
        ),
        "estimated_messages": len(estimated),
        "ground_truth_messages": len(ground_truth),
        "invalid_estimated_messages": invalid_estimated,
        "invalid_ground_truth_messages": invalid_ground_truth,
        "raw_packet_messages": len(packets),
        "invalid_packet_messages": invalid_packets,
        "evaluated_pairs": len(records),
        "skipped": skipped,
        "estimated_span_seconds": estimated_span_seconds,
        "effective_estimated_pose_rate_hz": pose_rate_hz(
            sample[0] for sample in estimated
        ),
        "translation_error_m": describe(translation_errors),
        "rotation_error_deg": describe(rotation_errors),
        "squared_translation_error_m2": describe(
            record["squared_translation_error_m2"]
            for record in records
        ),
        "squared_quaternion_error": describe(
            record["squared_quaternion_error"]
            for record in records
        ),
        "pose_loss": describe(
            record["pose_loss"] for record in records
        ),
        "ground_truth_sample_distance_ms": describe(
            record["ground_truth_sample_distance_ms"]
            for record in records
        ),
        "accuracy_thresholds": {
            "translation_m": args.translation_threshold_m,
            "rotation_deg": args.rotation_threshold_deg,
            "successful_pairs": threshold_successes,
            "success_rate_percent": (
                100.0 * threshold_successes / len(records)
            ),
        },
        "mean_signed_delta_xyz_m": [
            statistics.fmean(
                record["delta_xyz_m"][index]
                for record in records
            )
            for index in range(3)
        ],
        "estimated_xyz_ranges_m": coordinate_ranges(
            [sample[2] for sample in estimated]
        ),
        "matched_ground_truth_xyz_ranges_m": coordinate_ranges(
            used_ground_truth
        ),
        "tracking_telemetry": telemetry,
        "selected_marker": args.marker_label,
        "selected_marker_metrics": selected_marker_metrics,
        "runtime_timing": timing,
        "caveats": [
            (
                "Accuracy metrics use published poses only; "
                "missing poses are represented separately by "
                "tracking telemetry."
            ),
            (
                "Frames missed assumes the marker should be "
                "visible for the entire selected bag interval."
            ),
            (
                "processed_without_pose includes frames where "
                "a configured marker was not detected."
            ),
            (
                "Identical frame_id does not prove matching "
                "physical origins or axes."
            ),
            (
                "Header clocks must be synchronized; bag "
                "timestamps are receipt timestamps."
            ),
            (
                "image_time_ns uses the producer host-read "
                "timestamp rather than hardware exposure time."
            ),
        ],
    }

    return result, records


def discover_pose_topics(paths, storage):
    """List PoseStamped topics across all bag parts."""
    topics = set()

    if storage == "sqlite3":
        for path in paths:
            uri = path.resolve().as_uri() + "?mode=ro"

            with sqlite3.connect(uri, uri=True) as connection:
                topics.update(
                    name
                    for (name,) in connection.execute(
                        "SELECT name FROM topics WHERE type = ?",
                        (POSE_TYPE,),
                    )
                )
    else:
        rosbag2_py = import_rosbag2_py()

        for path in paths:
            reader = open_mcap_reader(path, rosbag2_py)
            topics.update(
                topic.name
                for topic in reader.get_all_topics_and_types()
                if topic.type == POSE_TYPE
            )

    return topics


def analyze(args):
    """Analyze each marker separately, or a legacy exact topic pair."""
    storage, paths = discover_bag(args.bag)
    topics = discover_pose_topics(paths, storage)

    def marker_topics(prefix):
        prefix = prefix.rstrip("/") + "/"

        return {
            topic[len(prefix):]: topic
            for topic in topics
            if topic.startswith(prefix)
            and topic[len(prefix):]
            and "/" not in topic[len(prefix):]
        }

    estimated_topics = marker_topics(args.estimated_topic)
    ground_truth_topics = marker_topics(args.ground_truth_topic)

    labels = sorted(
        estimated_topics.keys() | ground_truth_topics.keys()
    )

    if not labels:
        # Preserve support for legacy unsuffixed topics and explicitly
        # selected exact topics.
        result, records = analyze_topic_pair(args)

    else:
        per_bot = {}
        records = {}
        averages = {}

        for label in labels:
            estimated_topic = estimated_topics.get(label)
            ground_truth_topic = ground_truth_topics.get(label)

            if estimated_topic is None or ground_truth_topic is None:
                per_bot[label] = {
                    "estimated_topic": estimated_topic,
                    "ground_truth_topic": ground_truth_topic,
                    "error": "Missing estimated or ground-truth topic",
                }
                continue

            pair_args = argparse.Namespace(**vars(args))
            pair_args.estimated_topic = estimated_topic
            pair_args.ground_truth_topic = ground_truth_topic

            try:
                summary, samples = analyze_topic_pair(pair_args)
            except ValueError as error:
                # One bot with invalid or unmatched data does not
                # prevent valid bots from being analyzed.
                per_bot[label] = {
                    "estimated_topic": estimated_topic,
                    "ground_truth_topic": ground_truth_topic,
                    "error": str(error),
                }
                continue

            per_bot[label] = summary
            records[label] = samples

            averages[label] = {
                "evaluated_pairs": summary["evaluated_pairs"],
                "translation_error_m": (
                    summary["translation_error_m"]["mean"]
                ),
                "rotation_error_deg": (
                    summary["rotation_error_deg"]["mean"]
                ),
                "pose_loss": summary["pose_loss"]["mean"],
            }

        result = {
            "storage_identifier": storage,
            "bag_files": [str(path) for path in paths],
            "average_errors_by_bot": averages,
            "per_bot": per_bot,
        }

    if args.per_sample_output:
        args.per_sample_output.write_text(
            json.dumps(records, indent=2) + "\n",
            encoding="utf-8",
        )

    return result


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "bag",
        type=Path,
        help="A .db3/.mcap file or individual rosbag directory",
    )
    parser.add_argument(
        "--estimated-topic",
        default="/relative_pose/estimated",
        help="Exact pose topic or prefix containing per-bot pose topics",
    )
    parser.add_argument(
        "--ground-truth-topic",
        default="/relative_pose/ground_truth",
        help="Exact pose topic or prefix containing per-bot pose topics",
    )
    parser.add_argument(
        "--packet-topic",
        default="/mpose/packets",
    )
    parser.add_argument(
        "--marker-label",
        help=(
            "Marker whose missed-frame and reset metrics should "
            "be selected from telemetry; does not filter pose topics"
        ),
    )
    parser.add_argument(
        "--clock",
        choices=["header", "bag"],
        default="header",
    )
    parser.add_argument(
        "--method",
        choices=["interpolate", "nearest"],
        default="interpolate",
    )
    parser.add_argument(
        "--max-gap-ms",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--estimate-time-offset-ms",
        type=float,
        default=0.0,
        help=(
            "Add this offset to estimated timestamps for matching "
            "only; requires independent justification"
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--translation-threshold-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--rotation-threshold-deg",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional summary JSON output",
    )
    parser.add_argument(
        "--per-sample-output",
        type=Path,
        help=(
            "Optional per-pair JSON output, grouped by bot "
            "for marker topics"
        ),
    )

    args = parser.parse_args()

    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("beta must be finite and nonnegative")

    if not math.isfinite(args.max_gap_ms) or args.max_gap_ms <= 0:
        parser.error("max-gap-ms must be finite and positive")

    if not math.isfinite(args.estimate_time_offset_ms):
        parser.error("estimate-time-offset-ms must be finite")

    if (
        not math.isfinite(args.translation_threshold_m)
        or args.translation_threshold_m <= 0
    ):
        parser.error(
            "translation-threshold-m must be finite and positive"
        )

    if (
        not math.isfinite(args.rotation_threshold_deg)
        or args.rotation_threshold_deg <= 0
    ):
        parser.error(
            "rotation-threshold-deg must be finite and positive"
        )

    return args


def main():
    """Run bag analysis."""
    args = parse_arguments()

    try:
        result = analyze(args)
    except (
        ValueError,
        sqlite3.Error,
        OSError,
        UnicodeDecodeError,
        struct.error,
    ) as error:
        raise SystemExit(f"Error: {error}") from error

    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")

    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()