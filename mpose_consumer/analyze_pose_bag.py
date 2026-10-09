#!/usr/bin/env python3
"""Compare two PoseStamped topics in SQLite or MCAP ROS 2 bags.

SQLite input requires only Python 3.9+ and the standard library. MCAP input
requires ROS 2's rosbag2_py package and the MCAP storage plugin.

Input may be:

  - One .db3 file
  - One .mcap file
  - A rosbag directory containing .db3 files
  - A rosbag directory containing .mcap files

All storage parts in a directory are read. metadata.yaml is not required.

Defaults: header timestamps, linear translation interpolation, and
shortest-path quaternion SLERP. Both ground-truth bracket samples must lie
within --max-gap-ms.

Translation error:
  ||t_est - t_gt||

Rotation error:
  2 * acos(|dot(normalized(q_est), normalized(q_gt))|)

Pose loss:
  translation_error**2
  + beta * min(||q_est-q_gt||**2, ||q_est+q_gt||**2)

Quaternion order is ROS (x, y, z, w). No frame alignment is fitted. Both topics
must use one identical, nonempty frame_id. Only published, finite estimates are
evaluated; missing estimates are not assigned a loss.

Examples:

  python3 analyze_pose_bag.py recording.db3
  python3 analyze_pose_bag.py recording.mcap
  python3 analyze_pose_bag.py recording_directory --output summary.json
  python3 analyze_pose_bag.py recording_directory --method nearest --beta 10
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
SERIALIZATION_FORMAT = "cdr"


def decode_pose(data):
    """Decode a standard CDR v1 geometry_msgs/msg/PoseStamped message."""
    data = bytes(data)

    if len(data) < 4 or data[:2] not in (b"\x00\x01", b"\x00\x00"):
        raise ValueError("Expected standard CDR v1 encapsulation")

    order = "<" if data[1] == 1 else ">"
    payload = memoryview(data)[4:]

    if len(payload) < 12:
        raise ValueError("PoseStamped payload is too short")

    sec, nanosec, length = struct.unpack_from(order + "iII", payload, 0)

    if nanosec >= 1_000_000_000:
        raise ValueError("Invalid nanosecond value in message header")

    if length < 1 or 12 + length > len(payload):
        raise ValueError("Invalid frame_id string length")

    frame_bytes = bytes(payload[12 : 12 + length])

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


def normalize(q):
    """Return a normalized quaternion."""
    norm = math.sqrt(sum(value * value for value in q))

    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")

    return tuple(value / norm for value in q)


def add_sample(samples, data, bag_ns, clock):
    """Decode, validate, and append one PoseStamped sample."""
    header_ns, frame, translation, quaternion = decode_pose(data)
    stamp = header_ns if clock == "header" else bag_ns

    values = (*translation, *quaternion)

    if stamp <= 0 or not all(math.isfinite(value) for value in values):
        return False

    try:
        quaternion = normalize(quaternion)
    except ValueError:
        return False

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
    return True


def finish_topic(samples, invalid, found, name):
    """Sort and validate all samples collected for one topic."""
    if not found:
        raise ValueError(f"Topic not found: {name}")

    samples.sort(key=lambda sample: sample[0])

    duplicates = len(samples) - len({sample[0] for sample in samples})

    if duplicates:
        raise ValueError(f"{name}: {duplicates} duplicate timestamps; " "resolve stale or duplicate publications before evaluation")

    if not samples:
        raise ValueError(f"No valid samples: {name}")

    return samples, invalid


def read_sqlite_topic(paths, name, clock):
    """Read one topic from one or more SQLite rosbag parts."""
    samples = []
    invalid = 0
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
                (name,),
            ).fetchone()

            if topic is None:
                continue

            found = True
            topic_id, topic_type, serialization_format = topic
            actual = (topic_type, serialization_format)
            expected = (POSE_TYPE, SERIALIZATION_FORMAT)

            if actual != expected:
                raise ValueError(f"{name}: expected CDR PoseStamped, found {actual}")

            messages = connection.execute(
                """
                SELECT timestamp, data
                FROM messages
                WHERE topic_id = ?
                ORDER BY timestamp
                """,
                (topic_id,),
            )

            for bag_ns, data in messages:
                if not add_sample(samples, data, bag_ns, clock):
                    invalid += 1

    return finish_topic(samples, invalid, found, name)


def import_rosbag2_py():
    """Import rosbag2_py only when MCAP input is requested."""
    try:
        import rosbag2_py
    except ImportError as exc:
        raise ValueError("MCAP input requires ROS 2 rosbag2_py and the MCAP storage " "plugin. Source your ROS installation before running this script.") from exc

    return rosbag2_py


def open_mcap_reader(path, rosbag2_py):
    """Open one MCAP file with the ROS 2 sequential reader."""
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(path.resolve()),
        storage_id="mcap",
    )
    converter_options = rosbag2_py.ConverterOptions("", "")

    try:
        reader.open(storage_options, converter_options)
    except RuntimeError as exc:
        raise ValueError(f"Cannot open MCAP file {path}: {exc}") from exc

    return reader


def read_mcap_topic(paths, name, clock):
    """Read one topic from one or more MCAP rosbag parts."""
    rosbag2_py = import_rosbag2_py()

    samples = []
    invalid = 0
    found = False

    for path in paths:
        reader = open_mcap_reader(path, rosbag2_py)

        topics = {topic.name: topic for topic in reader.get_all_topics_and_types()}

        topic = topics.get(name)

        if topic is None:
            continue

        found = True
        actual = (topic.type, topic.serialization_format)
        expected = (POSE_TYPE, SERIALIZATION_FORMAT)

        if actual != expected:
            raise ValueError(f"{name}: expected CDR PoseStamped, found {actual}")

        try:
            while reader.has_next():
                topic_name, data, bag_ns = reader.read_next()

                if topic_name != name:
                    continue

                if not add_sample(samples, data, bag_ns, clock):
                    invalid += 1
        except RuntimeError as exc:
            raise ValueError(f"Error while reading MCAP file {path}: {exc}") from exc

    return finish_topic(samples, invalid, found, name)


def discover_bag(path):
    """Determine the storage type and storage files for a bag."""
    if path.is_file():
        suffix = path.suffix.lower()

        if suffix == ".db3":
            return "sqlite3", [path]

        if suffix == ".mcap":
            return "mcap", [path]

        raise ValueError("Supply a .db3 file, .mcap file, or rosbag directory")

    if not path.is_dir():
        raise ValueError("Supply an existing .db3 file, .mcap file, or rosbag directory")

    sqlite_paths = sorted(candidate for candidate in path.glob("*.db3") if candidate.is_file())
    mcap_paths = sorted(candidate for candidate in path.glob("*.mcap") if candidate.is_file())

    if sqlite_paths and mcap_paths:
        raise ValueError("Bag directory contains both .db3 and .mcap files; " "supply a directory containing only one storage format")

    if sqlite_paths:
        return "sqlite3", sqlite_paths

    if mcap_paths:
        return "mcap", mcap_paths

    raise ValueError("Bag directory contains no .db3 or .mcap files")


def read_topic(paths, storage, name, clock):
    """Read a topic using the selected storage backend."""
    if storage == "sqlite3":
        return read_sqlite_topic(paths, name, clock)

    if storage == "mcap":
        return read_mcap_topic(paths, name, clock)

    raise ValueError(f"Unsupported storage format: {storage}")


def slerp(a, b, fraction):
    """Perform shortest-path spherical interpolation between quaternions."""
    dot = sum(x * y for x, y in zip(a, b))

    if dot < 0:
        b = tuple(-value for value in b)
        dot = -dot

    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        interpolated = tuple((1 - fraction) * x + fraction * y for x, y in zip(a, b))
        return normalize(interpolated)

    angle = math.acos(dot)
    denominator = math.sin(angle)
    left_weight = math.sin((1 - fraction) * angle) / denominator
    right_weight = math.sin(fraction * angle) / denominator

    return normalize(tuple(left_weight * x + right_weight * y for x, y in zip(a, b)))


def percentile(values, proportion):
    """Calculate a linearly interpolated percentile."""
    values = sorted(values)
    index = (len(values) - 1) * proportion
    low = math.floor(index)
    high = math.ceil(index)

    return values[low] + (values[high] - values[low]) * (index - low)


def describe(values):
    """Return summary statistics for a sequence of numeric values."""
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "rmse": math.sqrt(statistics.fmean(value * value for value in values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def coordinate_ranges(samples):
    """Return the minimum and maximum for x, y, and z."""
    return [
        [
            min(sample[index] for sample in samples),
            max(sample[index] for sample in samples),
        ]
        for index in range(3)
    ]


def analyze(args):
    """Analyze the selected rosbag."""
    storage, paths = discover_bag(args.bag)

    estimated, invalid_estimated = read_topic(
        paths,
        storage,
        args.estimated_topic,
        args.clock,
    )
    ground_truth, invalid_ground_truth = read_topic(
        paths,
        storage,
        args.ground_truth_topic,
        args.clock,
    )

    estimated_frames = {sample[1] for sample in estimated}
    ground_truth_frames = {sample[1] for sample in ground_truth}

    if len(estimated_frames) != 1 or estimated_frames != ground_truth_frames or "" in estimated_frames:
        raise ValueError("Expected one identical nonempty frame_id: " f"estimated={estimated_frames}, " f"ground_truth={ground_truth_frames}")

    ground_truth_stamps = [sample[0] for sample in ground_truth]

    max_gap_ns = round(args.max_gap_ms * 1e6)
    offset_ns = round(args.estimate_time_offset_ms * 1e6)

    records = []
    used_ground_truth = []
    skipped = {
        "outside_ground_truth_range": 0,
        "gap_too_large": 0,
    }

    for sample in estimated:
        stamp, _, estimated_t, estimated_q, _, _ = sample
        matching_stamp = stamp + offset_ns

        if matching_stamp < ground_truth_stamps[0] or matching_stamp > ground_truth_stamps[-1]:
            skipped["outside_ground_truth_range"] += 1
            continue

        right = bisect.bisect_left(
            ground_truth_stamps,
            matching_stamp,
        )

        if ground_truth_stamps[right] == matching_stamp:
            ground_truth_t, ground_truth_q = ground_truth[right][2:4]
            gap_ns = 0

        elif args.method == "nearest":
            chosen = min(
                (right - 1, right),
                key=lambda index: abs(ground_truth_stamps[index] - matching_stamp),
            )
            gap_ns = abs(ground_truth_stamps[chosen] - matching_stamp)
            ground_truth_t, ground_truth_q = ground_truth[chosen][2:4]

        else:
            left = right - 1
            left_stamp = ground_truth_stamps[left]
            right_stamp = ground_truth_stamps[right]

            gap_ns = max(
                matching_stamp - left_stamp,
                right_stamp - matching_stamp,
            )

            fraction = (matching_stamp - left_stamp) / (right_stamp - left_stamp)

            ground_truth_t = tuple(
                (1 - fraction) * x + fraction * y
                for x, y in zip(
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
                estimated_t,
                ground_truth_t,
            )
        )

        translation_squared = sum(value * value for value in delta)

        quaternion_dot = abs(
            sum(
                estimated_value * ground_truth_value
                for estimated_value, ground_truth_value in zip(
                    estimated_q,
                    ground_truth_q,
                )
            )
        )
        quaternion_dot = min(1.0, max(0.0, quaternion_dot))

        quaternion_squared = min(
            sum(
                (estimated_value - ground_truth_value) ** 2
                for estimated_value, ground_truth_value in zip(
                    estimated_q,
                    ground_truth_q,
                )
            ),
            sum(
                (estimated_value + ground_truth_value) ** 2
                for estimated_value, ground_truth_value in zip(
                    estimated_q,
                    ground_truth_q,
                )
            ),
        )

        records.append(
            {
                "timestamp_ns": stamp,
                "matching_timestamp_ns": matching_stamp,
                "translation_error_m": math.sqrt(translation_squared),
                "rotation_error_deg": math.degrees(2 * math.acos(quaternion_dot)),
                "squared_translation_error_m2": translation_squared,
                "squared_quaternion_error": quaternion_squared,
                "pose_loss": (translation_squared + args.beta * quaternion_squared),
                "ground_truth_sample_distance_ms": gap_ns / 1e6,
                "delta_xyz_m": delta,
            }
        )

        used_ground_truth.append(ground_truth_t)

    if not records:
        raise ValueError(f"No matched samples. Skipped: {skipped}")

    def statistics_for(field):
        return describe([record[field] for record in records])

    result = {
        "storage_identifier": storage,
        "bag_files": [str(path) for path in paths],
        "estimated_topic": args.estimated_topic,
        "ground_truth_topic": args.ground_truth_topic,
        "frame_id": next(iter(estimated_frames)),
        "timestamp_source": args.clock,
        "matching_method": args.method,
        "maximum_sample_distance_ms": args.max_gap_ms,
        "estimate_time_offset_ms": args.estimate_time_offset_ms,
        "beta": args.beta,
        "pose_loss_definition": ("||t_est-t_gt||^2 + beta * " "min(||q_est-q_gt||^2, ||q_est+q_gt||^2), " "normalized quaternions"),
        "estimated_messages": len(estimated),
        "ground_truth_messages": len(ground_truth),
        "invalid_estimated_messages": invalid_estimated,
        "invalid_ground_truth_messages": invalid_ground_truth,
        "evaluated_pairs": len(records),
        "skipped": skipped,
        "estimated_span_seconds": (estimated[-1][0] - estimated[0][0]) / 1e9,
        "translation_error_m": statistics_for("translation_error_m"),
        "rotation_error_deg": statistics_for("rotation_error_deg"),
        "squared_translation_error_m2": statistics_for("squared_translation_error_m2"),
        "squared_quaternion_error": statistics_for("squared_quaternion_error"),
        "pose_loss": statistics_for("pose_loss"),
        "ground_truth_sample_distance_ms": statistics_for("ground_truth_sample_distance_ms"),
        "mean_signed_delta_xyz_m": [statistics.fmean(record["delta_xyz_m"][index] for record in records) for index in range(3)],
        "estimated_xyz_ranges_m": coordinate_ranges([sample[2] for sample in estimated]),
        "matched_ground_truth_xyz_ranges_m": coordinate_ranges(used_ground_truth),
        "caveats": [
            ("Published poses only; missing detections are not " "assigned a loss."),
            ("Identical frame_id does not prove matching physical " "origins or axes."),
            ("No offset or rotation is fitted to reduce the " "reported errors."),
            ("Header clocks must be synchronized; bag timestamps " "are receipt times."),
        ],
    }

    if args.per_sample_output:
        args.per_sample_output.write_text(json.dumps(records, indent=2) + "\n")

    return result


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "bag",
        type=Path,
        help="A .db3/.mcap file or rosbag directory",
    )
    parser.add_argument(
        "--estimated-topic",
        default="/relative_pose/estimated",
    )
    parser.add_argument(
        "--ground-truth-topic",
        default="/relative_pose/ground_truth",
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
        help=("Add this offset to estimated timestamps for matching only; " "requires independent justification"),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional summary JSON output path",
    )
    parser.add_argument(
        "--per-sample-output",
        type=Path,
        help="Optional per-pair JSON output path",
    )

    args = parser.parse_args()

    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("beta must be finite and nonnegative")

    if not math.isfinite(args.max_gap_ms) or args.max_gap_ms <= 0:
        parser.error("max-gap-ms must be finite and positive")

    if not math.isfinite(args.estimate_time_offset_ms):
        parser.error("estimate-time-offset-ms must be finite")

    return args


def main():
    """Run the command-line application."""
    args = parse_arguments()

    try:
        result = analyze(args)
    except (
        ValueError,
        sqlite3.Error,
        OSError,
        UnicodeDecodeError,
        struct.error,
    ) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")

    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
