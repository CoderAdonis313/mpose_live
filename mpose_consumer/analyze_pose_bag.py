#!/usr/bin/env python3
"""Analyze relative poses in a ROS 2 SQLite or MCAP bag.

Default inputs:
  /relative_pose/estimated/<bot>/<bot>
  /relative_pose/ground_truth/<bot>/<bot>

TransformStamped samples are matched by header.frame_id (arena) AND
child_frame_id (robot). Legacy PoseStamped and <prefix>/<bot> topics are
also supported. Errors are calculated here; gt_error_node outputs are
never used by the default topic selection.

Optional runtime telemetry comes only from /mpose/packets.
Translation is in metres; geodesic rotation error is in degrees.
"""

import argparse
import bisect
from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct


POSE_TYPE = "geometry_msgs/msg/PoseStamped"
TRANSFORM_TYPE = "geometry_msgs/msg/TransformStamped"
POSE_TYPES = {POSE_TYPE, TRANSFORM_TYPE}
STRING_TYPE = "std_msgs/msg/String"


def cdr_payload(data):
    data = bytes(data)
    if len(data) < 4 or data[:2] not in (b"\x00\x00", b"\x00\x01"):
        raise ValueError("Expected standard CDR v1 encapsulation")
    return ("<" if data[1] == 1 else ">"), memoryview(data)[4:]


def cdr_string(order, payload, offset):
    offset += (-offset) % 4
    length = struct.unpack_from(order + "I", payload, offset)[0]
    offset += 4
    if length < 1 or offset + length > len(payload):
        raise ValueError("Invalid CDR string length")
    value = bytes(payload[offset:offset + length])
    if value[-1:] != b"\x00":
        raise ValueError("CDR string is not null terminated")
    return value[:-1].decode("utf-8"), offset + length


def normalize(quaternion):
    norm = math.hypot(*quaternion)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")
    return tuple(value / norm for value in quaternion)


def decode_pose(data, message_type, legacy_bot):
    order, payload = cdr_payload(data)
    sec, nanosec = struct.unpack_from(order + "iI", payload, 0)
    if nanosec >= 1_000_000_000:
        raise ValueError("Invalid nanosecond value")
    arena, offset = cdr_string(order, payload, 8)
    if message_type == TRANSFORM_TYPE:
        bot, offset = cdr_string(order, payload, offset)
    elif message_type == POSE_TYPE:
        bot = legacy_bot
    else:
        raise ValueError(f"Unsupported pose type: {message_type}")
    offset += (-offset) % 8
    values = struct.unpack_from(order + "7d", payload, offset)
    if not arena or not bot:
        raise ValueError("Empty arena or robot frame")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite pose")
    return (
        sec * 1_000_000_000 + nanosec,
        arena, bot, values[:3], normalize(values[3:]),
    )


def discover_bag(path):
    if path.is_file() and path.suffix.lower() in (".db3", ".mcap"):
        return (
            "sqlite3" if path.suffix.lower() == ".db3" else "mcap",
            [path],
        )
    if not path.is_dir():
        raise ValueError("Supply a .db3/.mcap file or a rosbag directory")
    sqlite_paths = sorted(path.glob("*.db3"))
    mcap_paths = sorted(path.glob("*.mcap"))
    if sqlite_paths and mcap_paths:
        raise ValueError("Bag directory contains both SQLite and MCAP")
    if sqlite_paths:
        return "sqlite3", sqlite_paths
    if mcap_paths:
        return "mcap", mcap_paths
    raise ValueError("Bag directory contains no .db3 or .mcap files")


class Bag:
    """Read only selected topics, including across split bag files."""

    def __init__(self, path):
        self.storage, self.paths = discover_bag(path)
        self.topics = {}
        self.parts = []
        self.rosbag = None
        if self.storage == "mcap":
            try:
                import rosbag2_py
            except ImportError as error:
                raise ValueError(
                    "MCAP requires rosbag2_py and its MCAP storage plugin; "
                    "source your ROS environment first"
                ) from error
            self.rosbag = rosbag2_py
        for part in self.paths:
            if self.storage == "sqlite3":
                with self.connect(part) as connection:
                    metadata = connection.execute(
                        "SELECT name, type, serialization_format FROM topics"
                    ).fetchall()
            else:
                reader = self.open_mcap(part)
                metadata = [
                    (topic.name, topic.type, topic.serialization_format)
                    for topic in reader.get_all_topics_and_types()
                ]
            part_topics = {}
            for name, message_type, encoding in metadata:
                signature = (message_type, encoding)
                part_topics.setdefault(name, set()).add(signature)
                self.topics.setdefault(name, set()).add(signature)
            self.parts.append((part, part_topics))

    @staticmethod
    def connect(path):
        return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)

    def open_mcap(self, path):
        reader = self.rosbag.SequentialReader() # type: ignore
        reader.open(
            self.rosbag.StorageOptions( # type: ignore
                uri=str(path.resolve()), storage_id="mcap"
            ),
            self.rosbag.ConverterOptions("", ""), # type: ignore
        )
        return reader

    def topic_type(self, name, allowed):
        signatures = self.topics.get(name)
        if not signatures:
            raise ValueError(f"Topic not found: {name}")
        if len(signatures) != 1:
            raise ValueError(f"{name}: inconsistent bag metadata: {signatures}")
        message_type, encoding = next(iter(signatures))
        if message_type not in allowed or encoding != "cdr":
            raise ValueError(
                f"{name}: expected CDR {sorted(allowed)}, "
                f"found {(message_type, encoding)}"
            )
        return message_type

    def read(self, names):
        messages = {name: [] for name in names}
        for path, metadata in self.parts:
            selected = sorted(set(names) & metadata.keys())
            if not selected:
                continue
            if self.storage == "sqlite3":
                placeholders = ",".join("?" for _ in selected)
                with self.connect(path) as connection:
                    rows = connection.execute(
                        "SELECT t.name, m.timestamp, m.data "
                        "FROM messages m JOIN topics t ON m.topic_id = t.id "
                        f"WHERE t.name IN ({placeholders}) "
                        "ORDER BY m.timestamp",
                        selected,
                    )
                    for name, stamp, data in rows:
                        messages[name].append((stamp, data))
            else:
                reader = self.open_mcap(path)
                reader.set_filter(self.rosbag.StorageFilter(topics=selected)) # type: ignore
                while reader.has_next():
                    name, data, stamp = reader.read_next()
                    if name in messages:
                        messages[name].append((stamp, data))
        for rows in messages.values():
            rows.sort(key=lambda row: row[0])
        return messages


def select_topics(bag, selector, exact_label):
    """Accept an exact topic, <prefix>/<bot>, or <prefix>/<bot>/<bot>."""
    selector = selector.rstrip("/")
    if selector in bag.topics:
        bag.topic_type(selector, POSE_TYPES)
        return {selector: exact_label}
    prefix = selector + "/"
    selected = {}
    for name in sorted(bag.topics):
        if not name.startswith(prefix):
            continue
        parts = name[len(prefix):].split("/")
        if (
            (len(parts) == 1 and parts[0])
            or (len(parts) == 2 and parts[0] and parts[0] == parts[1])
        ):
            bag.topic_type(name, POSE_TYPES)
            selected[name] = parts[-1]
    return selected


def read_groups(bag, topics, messages, args):
    groups = defaultdict(list)
    sources = defaultdict(set)
    counts = {}
    for topic, legacy_bot in topics.items():
        message_type = bag.topic_type(topic, POSE_TYPES)
        invalid = filtered = 0
        for bag_ns, data in messages[topic]:
            try:
                header_ns, arena, bot, translation, quaternion = decode_pose(
                    data, message_type, legacy_bot
                )
                stamp = header_ns if args.clock == "header" else bag_ns
                if stamp <= 0:
                    raise ValueError("Invalid timestamp")
            except (ValueError, UnicodeError, struct.error):
                invalid += 1
                continue
            if (
                (args.arena_frame and arena != args.arena_frame)
                or (args.bot_frame and bot != args.bot_frame)
            ):
                filtered += 1
                continue
            key = (arena, bot)
            groups[key].append(
                (stamp, arena, translation, quaternion, bag_ns, header_ns)
            )
            sources[key].add(topic)
        counts[topic] = {
            "message_type": message_type,
            "messages": len(messages[topic]),
            "invalid_messages": invalid,
            "filtered_messages": filtered,
        }
    for samples in groups.values():
        samples.sort(key=lambda sample: sample[0])
    return groups, sources, counts


def describe(values):
    values = sorted(float(value) for value in values if math.isfinite(value))
    if not values:
        return None

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "rmse": math.sqrt(statistics.fmean(value * value for value in values)),
        "minimum": values[0],
        "maximum": values[-1],
    }


def pose_rate_hz(stamps):
    stamps = sorted(set(stamps))
    if len(stamps) < 2:
        return None
    return (len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])


def slerp(first, second, fraction):
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0:
        second = tuple(-value for value in second)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize(tuple(
            (1 - fraction) * a + fraction * b
            for a, b in zip(first, second)
        ))
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return normalize(tuple(
        (math.sin((1 - fraction) * angle) * a
         + math.sin(fraction * angle) * b) / denominator
        for a, b in zip(first, second)
    ))


def match_samples(estimated, ground_truth, args):
    stamps = [sample[0] for sample in ground_truth]
    max_gap = round(args.max_gap_ms * 1e6)
    offset = round(args.estimate_time_offset_ms * 1e6)
    records, used_positions = [], []
    skipped = {"outside_ground_truth_range": 0, "gap_too_large": 0}
    for stamp, arena, estimated_t, estimated_q, _, _ in estimated:
        target = stamp + offset
        if target < stamps[0] or target > stamps[-1]:
            skipped["outside_ground_truth_range"] += 1
            continue
        right = bisect.bisect_left(stamps, target)
        if stamps[right] == target:
            ground_t, ground_q = ground_truth[right][2:4]
            gap = 0
        elif args.method == "nearest":
            chosen = min((right - 1, right), key=lambda i: abs(stamps[i] - target))
            ground_t, ground_q = ground_truth[chosen][2:4]
            gap = abs(stamps[chosen] - target)
        else:
            left = right - 1
            gap = max(target - stamps[left], stamps[right] - target)
            fraction = (target - stamps[left]) / (stamps[right] - stamps[left])
            ground_t = tuple(
                (1 - fraction) * a + fraction * b
                for a, b in zip(ground_truth[left][2], ground_truth[right][2])
            )
            ground_q = slerp(
                ground_truth[left][3], ground_truth[right][3], fraction
            )
        if gap > max_gap:
            skipped["gap_too_large"] += 1
            continue
        delta = tuple(a - b for a, b in zip(estimated_t, ground_t))
        translation_squared = sum(value * value for value in delta)
        translation_error = math.sqrt(translation_squared)
        dot = abs(sum(a * b for a, b in zip(estimated_q, ground_q)))
        rotation_error = math.degrees(2 * math.acos(min(1.0, dot)))
        quaternion_squared = min(
            sum((a - b) ** 2 for a, b in zip(estimated_q, ground_q)),
            sum((a + b) ** 2 for a, b in zip(estimated_q, ground_q)),
        )
        records.append({
            "timestamp_ns": stamp,
            "matching_timestamp_ns": target,
            "frame_id": arena,
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
            "squared_translation_error_m2": translation_squared,
            "squared_quaternion_error": quaternion_squared,
            "pose_loss": translation_error + args.beta * math.sqrt(quaternion_squared),
            "ground_truth_sample_distance_ms": gap / 1e6,
            "delta_xyz_m": delta,
            "within_accuracy_thresholds": (
                translation_error <= args.translation_threshold_m
                and rotation_error <= args.rotation_threshold_deg
            ),
        })
        used_positions.append(ground_t)
    return records, used_positions, skipped


def coordinate_ranges(positions):
    return [
        [min(position[i] for position in positions),
         max(position[i] for position in positions)]
        for i in range(3)
    ]


def analyze_group(estimated, ground_truth, args):
    for name, samples in (("estimated", estimated), ("ground_truth", ground_truth)):
        if not samples:
            raise ValueError(f"No {name} samples for this arena/robot")
        duplicates = len(samples) - len({sample[0] for sample in samples})
        if duplicates:
            raise ValueError(
                f"{name}: {duplicates} duplicate timestamps within this arena/robot"
            )
    records, used_positions, skipped = match_samples(estimated, ground_truth, args)
    if not records:
        raise ValueError(f"No matched samples; skipped={skipped}")
    successes = sum(record["within_accuracy_thresholds"] for record in records)
    metrics = (
        "translation_error_m", "rotation_error_deg", "pose_loss",
        "squared_translation_error_m2", "squared_quaternion_error",
        "ground_truth_sample_distance_ms",
    )
    summary = {
        "estimated_messages": len(estimated),
        "ground_truth_messages": len(ground_truth),
        "evaluated_pairs": len(records),
        "skipped": skipped,
        "estimated_span_seconds": (estimated[-1][0] - estimated[0][0]) / 1e9,
        "effective_estimated_pose_rate_hz": pose_rate_hz(s[0] for s in estimated),
        **{metric: describe(record[metric] for record in records) for metric in metrics},
        "accuracy_thresholds": {
            "translation_m": args.translation_threshold_m,
            "rotation_deg": args.rotation_threshold_deg,
            "successful_pairs": successes,
            "success_rate_percent": 100.0 * successes / len(records),
        },
        "mean_signed_delta_xyz_m": [
            statistics.fmean(record["delta_xyz_m"][i] for record in records)
            for i in range(3)
        ],
        "estimated_xyz_ranges_m": coordinate_ranges([s[2] for s in estimated]),
        "matched_ground_truth_xyz_ranges_m": coordinate_ranges(used_positions),
    }
    return summary, records


def read_packets(rows):
    packets, invalid = [], 0
    for bag_ns, data in rows:
        try:
            order, payload = cdr_payload(data)
            value, _ = cdr_string(order, payload, 0)
            packet = json.loads(value)
            if not isinstance(packet, dict) or packet.get("schema_version") != 1:
                raise ValueError("Unsupported packet schema")
            if not isinstance(packet.get("session_id"), str) or not packet["session_id"]:
                raise ValueError("Invalid session")
            if type(packet.get("packet_seq")) is not int or packet["packet_seq"] < 0:
                raise ValueError("Invalid packet sequence")
            if type(packet.get("send_time_ns")) is not int or packet["send_time_ns"] <= 0:
                raise ValueError("Invalid send timestamp")
        except (ValueError, UnicodeError, struct.error):
            invalid += 1
            continue
        packets.append((bag_ns, packet))
    return packets, invalid


def summarize_telemetry(packets):
    scalar_fields = (
        "submitted_inputs", "processed_inputs", "dropped_inputs",
        "manual_resets", "camera_frames_skipped",
    )
    marker_fields = (
        "automatic_resets", "processed_without_pose",
        "detected_without_pose", "dropped_detected_inputs",
    )
    sessions = {}
    for _, packet in packets:
        telemetry = packet.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        session = sessions.setdefault(
            packet["session_id"],
            {"scalars": {}, "markers": {field: {} for field in marker_fields}},
        )
        for field in scalar_fields:
            value = telemetry.get(field)
            if type(value) is int and value >= 0:
                session["scalars"][field] = max(session["scalars"].get(field, 0), value)
        for field in marker_fields:
            values = telemetry.get(field)
            if not isinstance(values, dict):
                continue
            for label, value in values.items():
                if isinstance(label, str) and type(value) is int and value >= 0:
                    counters = session["markers"][field]
                    counters[label] = max(counters.get(label, 0), value)
    if not sessions:
        return {"available": False, "session_count": 0}
    totals = {
        field: sum(session["scalars"].get(field, 0) for session in sessions.values())
        for field in scalar_fields
    }
    markers = {field: defaultdict(int) for field in marker_fields}
    for session in sessions.values():
        for field in marker_fields:
            for label, value in session["markers"][field].items():
                markers[field][label] += value
    eligible = totals["camera_frames_skipped"] + totals["submitted_inputs"]
    per_marker = {}
    for label in sorted(set().union(*(set(values) for values in markers.values()))):
        without_pose = markers["processed_without_pose"].get(label, 0)
        successful = max(0, totals["processed_inputs"] - without_pose)
        missed = max(0, eligible - successful)
        resets = markers["automatic_resets"].get(label, 0)
        per_marker[label] = {
            "successful_pose_frames": successful,
            "frames_missed_assuming_visible": missed,
            "frames_missed_percentage_assuming_visible": (
                100.0 * missed / eligible if eligible else None
            ),
            "automatic_resets": resets,
            "manual_global_resets": totals["manual_resets"],
            "total_resets_affecting_marker": resets + totals["manual_resets"],
            "processed_without_pose": without_pose,
            "detected_without_pose": markers["detected_without_pose"].get(label, 0),
            "dropped_detected_inputs": markers["dropped_detected_inputs"].get(label, 0),
        }
    return {
        "available": True,
        "session_count": len(sessions),
        **totals,
        "eligible_camera_frames_assuming_visible": eligible,
        "per_marker": per_marker,
    }


def summarize_packet_timing(packets):
    unique = {}
    for bag_ns, packet in packets:
        frame = packet.get("frame_index")
        image_time = packet.get("image_time_ns")
        result_time = packet.get("result_time_ns")
        if (
            type(frame) is not int or type(image_time) is not int
            or type(result_time) is not int
            or image_time <= 0 or result_time < image_time
        ):
            continue
        key = (packet["session_id"], frame, image_time, result_time)
        previous = unique.get(key)
        if previous is None or (
            packet.get("valid") is True and previous[1].get("valid") is not True
        ):
            unique[key] = (bag_ns, packet)
    metrics = {
        "inference_ms": [], "image_to_result_ms": [],
        "image_to_send_ms": [], "send_to_bag_receipt_ms": [],
    }
    marker_stamps = defaultdict(list)
    for bag_ns, packet in unique.values():
        inference = packet.get("inference_seconds")
        if (
            isinstance(inference, (int, float))
            and math.isfinite(inference) and inference >= 0
        ):
            metrics["inference_ms"].append(inference * 1000)
        image_time, result_time = packet["image_time_ns"], packet["result_time_ns"]
        metrics["image_to_result_ms"].append((result_time - image_time) / 1e6)
        send_time = packet["send_time_ns"]
        if send_time >= result_time:
            metrics["image_to_send_ms"].append((send_time - image_time) / 1e6)
            metrics["send_to_bag_receipt_ms"].append((bag_ns - send_time) / 1e6)
        poses = packet.get("poses")
        if packet.get("valid") is True and isinstance(poses, dict):
            for label, pose in poses.items():
                if isinstance(label, str) and isinstance(pose, dict) and pose.get("valid") is True:
                    marker_stamps[label].append(image_time)
    return {
        "unique_inference_results": len(unique),
        **{name: describe(values) for name, values in metrics.items()},
        "effective_pose_rate_hz_by_marker": {
            label: pose_rate_hz(stamps) for label, stamps in sorted(marker_stamps.items())
        },
        "valid_unique_pose_results_by_marker": {
            label: len(set(stamps)) for label, stamps in sorted(marker_stamps.items())
        },
    }


def analyze(args):
    bag = Bag(args.bag)
    estimated_tail = args.estimated_topic.rstrip("/").split("/")[-1]
    ground_tail = args.ground_truth_topic.rstrip("/").split("/")[-1]
    exact_label = estimated_tail if estimated_tail == ground_tail else "legacy"
    estimated_topics = select_topics(bag, args.estimated_topic, exact_label)
    ground_topics = select_topics(bag, args.ground_truth_topic, exact_label)
    if not estimated_topics or not ground_topics:
        raise ValueError(
            "No matching relative-pose topics. Expected "
            f"{args.estimated_topic}/<bot>/<bot> and "
            f"{args.ground_truth_topic}/<bot>/<bot>, or explicit exact topics. "
            "Record both rel_pose_node outputs."
        )
    if set(estimated_topics) & set(ground_topics):
        raise ValueError("Estimated and ground-truth topic selections overlap")

    selected = set(estimated_topics) | set(ground_topics)
    packet_found = args.packet_topic in bag.topics
    if not args.no_telemetry and packet_found:
        bag.topic_type(args.packet_topic, {STRING_TYPE})
        selected.add(args.packet_topic)
    messages = bag.read(selected)
    estimated, estimated_sources, estimated_counts = read_groups(
        bag, estimated_topics, messages, args
    )
    ground, ground_sources, ground_counts = read_groups(
        bag, ground_topics, messages, args
    )
    if not estimated and not ground:
        raise ValueError(
            "No valid relative poses after filtering. "
            f"Estimated topics: {estimated_counts}; ground truth: {ground_counts}"
        )

    per_bot, averages, all_records = {}, {}, {}
    successful_groups = failed_groups = 0
    for arena, bot in sorted(estimated.keys() | ground.keys()):
        key = (arena, bot)
        summary = {
            "frame_id": arena,
            "child_frame_id": bot,
            "estimated_topics": sorted(estimated_sources.get(key, set())),
            "ground_truth_topics": sorted(ground_sources.get(key, set())),
        }
        try:
            metrics, records = analyze_group(
                estimated.get(key, []), ground.get(key, []), args
            )
        except ValueError as error:
            summary["error"] = str(error)
            failed_groups += 1
        else:
            successful_groups += 1
            summary.update(metrics)
            for record in records:
                record["child_frame_id"] = bot
            all_records.setdefault(bot, {})[arena] = records
            averages.setdefault(bot, {})[arena] = {
                "evaluated_pairs": summary["evaluated_pairs"],
                **{
                    metric: summary[metric]["mean"]
                    for metric in ("translation_error_m", "rotation_error_deg", "pose_loss")
                },
            }
        per_bot.setdefault(bot, {"per_arena": {}})["per_arena"][arena] = summary

    packets, invalid_packets = ([], 0)
    if not args.no_telemetry and packet_found:
        packets, invalid_packets = read_packets(messages[args.packet_topic])
    telemetry = summarize_telemetry(packets)
    selected_metrics = None
    if args.marker_label:
        selected_metrics = telemetry.get("per_marker", {}).get(args.marker_label)
        if selected_metrics is None:
            raise ValueError(f"No packet telemetry for marker {args.marker_label!r}")

    result = {
        "storage_identifier": bag.storage,
        "bag_files": [str(path) for path in bag.paths],
        "timestamp_source": args.clock,
        "matching_method": args.method,
        "maximum_sample_distance_ms": args.max_gap_ms,
        "estimate_time_offset_ms": args.estimate_time_offset_ms,
        "arena_frame_filter": args.arena_frame,
        "bot_frame_filter": args.bot_frame,
        "beta": args.beta,
        "pose_loss_definition": "||t_est-t_gt|| + beta * min(||q_est-q_gt||, ||q_est+q_gt||)",
        "successful_arena_bot_groups": successful_groups,
        "failed_arena_bot_groups": failed_groups,
        "average_errors_by_bot_and_arena": averages,
        "per_bot": per_bot,
        "pose_topic_counts": {
            "estimated": estimated_counts,
            "ground_truth": ground_counts,
        },
        "telemetry_enabled": not args.no_telemetry,
        "packet_topic": args.packet_topic,
        "packet_topic_found": packet_found,
        "raw_packet_messages": len(packets),
        "invalid_packet_messages": invalid_packets,
        "tracking_telemetry": telemetry,
        "runtime_timing": summarize_packet_timing(packets),
        "selected_marker": args.marker_label,
        "selected_marker_metrics": selected_metrics,
        "caveats": [
            "Errors are computed from relative poses; precomputed error topics are not used.",
            "Only identical arena and robot frame IDs are compared.",
            "Statistics remain separate for each arena/robot; samples are not pooled across arenas.",
            "Header clocks must be synchronized; bag timestamps are receipt timestamps.",
            "Nearest matching and interpolation do not extrapolate outside the ground-truth interval.",
            "Packet telemetry describes producer sessions, not individual arena/robot comparisons.",
            "Frames missed assumes continuous visibility; cumulative counters may predate bag recording.",
            "Identical frame names do not prove matching physical origins or axes.",
        ],
    }
    if args.per_sample_output:
        args.per_sample_output.write_text(
            json.dumps(all_records, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bag", type=Path, help="Bag directory, .db3 file, or .mcap file")
    parser.add_argument("--estimated-topic", default="/relative_pose/estimated",
                        help="Exact topic or prefix containing relative-pose topics")
    parser.add_argument("--ground-truth-topic", default="/relative_pose/ground_truth",
                        help="Exact topic or prefix containing relative-pose topics")
    parser.add_argument("--arena-frame", help="Only analyze this header.frame_id")
    parser.add_argument("--bot-frame", help="Only analyze this child_frame_id")
    parser.add_argument("--packet-topic", default="/mpose/packets")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="Read relative poses only; skip packet telemetry")
    parser.add_argument("--marker-label",
                        help="Select packet telemetry for a marker; does not filter poses")
    parser.add_argument("--clock", choices=("header", "bag"), default="header")
    parser.add_argument("--method", choices=("interpolate", "nearest"), default="interpolate")
    parser.add_argument("--max-gap-ms", type=float, default=100.0)
    parser.add_argument("--estimate-time-offset-ms", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--translation-threshold-m", type=float, default=0.05)
    parser.add_argument("--rotation-threshold-deg", type=float, default=10.0)
    parser.add_argument("--output", type=Path, help="Summary JSON")
    parser.add_argument("--per-sample-output", type=Path,
                        help="Per-sample JSON grouped by robot, then arena")
    args = parser.parse_args(argv)
    for name in ("max_gap_ms", "translation_threshold_m", "rotation_threshold_deg"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("--beta must be finite and nonnegative")
    if not math.isfinite(args.estimate_time_offset_ms):
        parser.error("--estimate-time-offset-ms must be finite")
    if args.no_telemetry and args.marker_label:
        parser.error("--marker-label cannot be used with --no-telemetry")
    return args


def main():
    args = parse_arguments()
    try:
        result = analyze(args)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
    except (ValueError, sqlite3.Error, OSError, RuntimeError, struct.error) as error:
        raise SystemExit(f"Error: {error}") from error
    print(text, end="")
    if result["successful_arena_bot_groups"] == 0:
        raise SystemExit("Error: no arena/robot group had matched samples")


if __name__ == "__main__":
    main()