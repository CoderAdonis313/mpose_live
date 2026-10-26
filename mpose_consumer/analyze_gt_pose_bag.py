#!/usr/bin/env python3
"""Summarize recorded gt_error_node PoseError messages.

Requires PoseError with these fields:
    Header, bot_label, translation_error_m, rotation_error_deg,
    rotation_error_quat, pose_loss, beta.

Reuses bag readers, statistics, and telemetry helpers from
analyze_pose_bag.py in the same directory.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3
import struct

from analyze_pose_bag import (
    Bag,
    STRING_TYPE,
    cdr_payload,
    cdr_string,
    describe,
    pose_rate_hz,
    read_packets,
    summarize_packet_timing,
    summarize_telemetry,
)


ERROR_TYPE = "mpose_interfaces/msg/PoseError"

METRICS = (
    "translation_error_m",
    "rotation_error_deg",
    "rotation_error_quat",
    "pose_loss",
    "squared_translation_error_m2",
    "squared_quaternion_error",
)

UNAVAILABLE = {
    "estimated_messages":
        "Source pose counts are not recorded in PoseError.",
    "ground_truth_messages":
        "Source pose counts are not recorded in PoseError.",
    "skipped":
        "Rejected/unmatched comparisons produce no PoseError message.",
    "estimated_span_seconds":
        "Only successfully published errors are available.",
    "effective_estimated_pose_rate_hz":
        "Error rate is not the input pose rate.",
    "ground_truth_sample_distance_ms":
        "Interpolation gaps are not recorded.",
    "mean_signed_delta_xyz_m":
        "PoseError contains a distance, not an XYZ delta.",
    "estimated_xyz_ranges_m":
        "Estimated positions are not recorded.",
    "matched_ground_truth_xyz_ranges_m":
        "Ground-truth positions are not recorded.",
}


def decode_error(data):
    """Decode the current PoseError CDR layout."""
    order, payload = cdr_payload(data)

    sec, nanosec = struct.unpack_from(order + "iI", payload, 0)
    if nanosec >= 1_000_000_000:
        raise ValueError("Invalid nanoseconds")

    arena, offset = cdr_string(order, payload, 8)
    bot, offset = cdr_string(order, payload, offset)

    # Align the following float64 fields to an 8-byte boundary.
    offset += (-offset) % 8

    # Five float64 fields, including beta.
    if len(payload) != offset + 40:
        raise ValueError(
            "Unexpected PoseError layout: expected five float64 "
            "fields including beta. Older bags without beta "
            "are not supported by this version."
        )

    translation, rotation, quaternion, loss, beta = struct.unpack_from(
        order + "5d", payload, offset
    )

    stamp = sec * 1_000_000_000 + nanosec
    if stamp <= 0 or not arena or not bot:
        raise ValueError("Missing timestamp, arena, or bot")

    if not all(
        math.isfinite(value) and value >= 0
        for value in (translation, rotation, quaternion, loss)
    ):
        raise ValueError("Nonfinite or negative error metric")

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("Beta must be finite and positive")

    squared_translation = translation * translation
    squared_quaternion = quaternion * quaternion

    if not all(
        math.isfinite(value)
        for value in (squared_translation, squared_quaternion)
    ):
        raise ValueError("Squared error overflow")

    return {
        "timestamp_ns": stamp,
        "frame_id": arena,
        "child_frame_id": bot,
        "translation_error_m": translation,
        "rotation_error_deg": rotation,
        "rotation_error_quat": quaternion,
        "pose_loss": loss,
        "squared_translation_error_m2": squared_translation,
        "squared_quaternion_error": squared_quaternion,
        "beta": beta,
    }


def select_error_topics(bag, selector):
    """Select an exact PoseError topic or topics below a prefix."""
    selector = selector.rstrip("/")

    if selector in bag.topics:
        bag.topic_type(selector, {ERROR_TYPE})
        return [selector]

    selected = []
    for topic, signatures in sorted(bag.topics.items()):
        if not topic.startswith(selector + "/"):
            continue

        # Ignore legacy Float64 error topics.
        if any(kind == ERROR_TYPE for kind, _ in signatures):
            bag.topic_type(topic, {ERROR_TYPE})
            selected.append(topic)

    if not selected:
        raise ValueError(
            f"No {ERROR_TYPE} topics below {selector}/"
        )

    return selected


def read_error_groups(topics, messages, args):
    """Decode, filter, and group records by arena and bot."""
    groups = defaultdict(list)
    sources = defaultdict(set)
    counts = {}

    for topic in topics:
        invalid = filtered = 0

        for bag_ns, data in messages[topic]:
            try:
                record = decode_error(data)
                stamp = (
                    record["timestamp_ns"]
                    if args.clock == "header"
                    else bag_ns
                )
                if stamp <= 0:
                    raise ValueError("Invalid selected timestamp")
            except (ValueError, UnicodeError, struct.error):
                invalid += 1
                continue

            arena = record["frame_id"]
            bot = record["child_frame_id"]

            if (
                (args.arena_frame and arena != args.arena_frame)
                or (args.bot_frame and bot != args.bot_frame)
            ):
                filtered += 1
                continue

            record["bag_timestamp_ns"] = bag_ns
            record["analysis_timestamp_ns"] = stamp
            record["source_topic"] = topic
            record["matching_timestamp_ns"] = record["timestamp_ns"]
            record["ground_truth_sample_distance_ms"] = None
            record["delta_xyz_m"] = None
            record["within_accuracy_thresholds"] = (
                record["translation_error_m"]
                <= args.translation_threshold_m
                and record["rotation_error_deg"]
                <= args.rotation_threshold_deg
            )

            groups[(arena, bot)].append(record)
            sources[(arena, bot)].add(topic)

        counts[topic] = {
            "message_type": ERROR_TYPE,
            "messages": len(messages[topic]),
            "invalid_messages": invalid,
            "filtered_messages": filtered,
        }

    for records in groups.values():
        records.sort(
            key=lambda record: record["analysis_timestamp_ns"]
        )

    return groups, sources, counts


def summarize_beta(records):
    """Report a single beta only when all supplied records agree."""
    values = sorted({record["beta"] for record in records})
    return {
        "beta": values[0] if len(values) == 1 else None,
        "beta_values": values,
        "beta_varies": len(values) > 1,
    }


def summarize_group(records, args):
    """Calculate statistics for one arena/bot pair."""
    # Check measurement timestamps even when receipt time is selected.
    duplicates = len(records) - len(
        {record["timestamp_ns"] for record in records}
    )
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate measurement timestamps "
            "within this arena/robot"
        )

    stamps = [
        record["analysis_timestamp_ns"]
        for record in records
    ]
    successes = sum(
        record["within_accuracy_thresholds"]
        for record in records
    )

    return {
        **{name: None for name in UNAVAILABLE},
        "unavailable_metrics": UNAVAILABLE,
        **summarize_beta(records),
        "evaluated_pairs": len(records),
        "recorded_error_messages": len(records),
        "error_span_seconds": (
            max(stamps) - min(stamps)
        ) / 1e9,
        "effective_error_rate_hz": pose_rate_hz(stamps),
        **{
            metric: describe(record[metric] for record in records)
            for metric in METRICS
        },
        "accuracy_thresholds": {
            "translation_m": args.translation_threshold_m,
            "rotation_deg": args.rotation_threshold_deg,
            "successful_pairs": successes,
            "success_rate_percent": (
                100.0 * successes / len(records)
            ),
        },
    }


def analyze(args):
    bag = Bag(args.bag)
    topics = select_error_topics(bag, args.error_topic)
    selected = set(topics)

    packet_found = args.packet_topic in bag.topics
    if not args.no_telemetry and packet_found:
        bag.topic_type(args.packet_topic, {STRING_TYPE})
        selected.add(args.packet_topic)

    messages = bag.read(selected)
    groups, sources, counts = read_error_groups(
        topics, messages, args
    )

    if not groups:
        raise ValueError(
            "No valid errors after filtering. Check that the bag "
            "contains the updated PoseError messages with beta. "
            f"Topic counts: {counts}"
        )

    per_bot = {}
    averages = {}
    all_records = {}
    successful_groups = failed_groups = 0

    for (arena, bot), records in sorted(groups.items()):
        summary = {
            "frame_id": arena,
            "child_frame_id": bot,
            "error_topics": sorted(sources[(arena, bot)]),
            "estimated_topics": None,
            "ground_truth_topics": None,
        }

        try:
            summary.update(summarize_group(records, args))
        except ValueError as error:
            summary["error"] = str(error)
            failed_groups += 1
        else:
            successful_groups += 1
            all_records.setdefault(bot, {})[arena] = records

            averages.setdefault(bot, {})[arena] = {
                "evaluated_pairs": len(records),
                "beta": summary["beta"],
                "beta_values": summary["beta_values"],
                "beta_varies": summary["beta_varies"],
                **{
                    metric: summary[metric]["mean"]
                    for metric in (
                        "translation_error_m",
                        "rotation_error_deg",
                        "pose_loss",
                    )
                },
            }

        per_bot.setdefault(
            bot, {"per_arena": {}}
        )["per_arena"][arena] = summary

    packets, invalid_packets = [], 0
    if not args.no_telemetry and packet_found:
        packets, invalid_packets = read_packets(
            messages[args.packet_topic]
        )

    telemetry = summarize_telemetry(packets)
    selected_metrics = None

    if args.marker_label:
        selected_metrics = telemetry.get(
            "per_marker", {}
        ).get(args.marker_label)

        if selected_metrics is None:
            raise ValueError(
                f"No packet telemetry for marker "
                f"{args.marker_label!r}"
            )

    # Overall beta metadata covers successfully analyzed groups only.
    beta_summary = summarize_beta(
        record
        for arenas in all_records.values()
        for records in arenas.values()
        for record in records
    )

    result = {
        "analysis_source": "recorded_gt_error_node",
        "storage_identifier": bag.storage,
        "bag_files": [str(path) for path in bag.paths],
        "timestamp_source": args.clock,
        "matching_method": "already_matched_by_gt_error_node",
        "maximum_sample_distance_ms": None,
        "estimate_time_offset_ms": None,
        "arena_frame_filter": args.arena_frame,
        "bot_frame_filter": args.bot_frame,
        **beta_summary,
        "pose_loss_source": (
            "recorded PoseError.pose_loss; not recomputed"
        ),
        "pose_loss_definition": (
            "||t_est-t_gt|| + beta * "
            "min(||q_est-q_gt||, ||q_est+q_gt||)"
        ),
        "successful_arena_bot_groups": successful_groups,
        "failed_arena_bot_groups": failed_groups,
        "average_errors_by_bot_and_arena": averages,
        "per_bot": per_bot,
        "error_topic_counts": counts,
        "pose_topic_counts": {
            "estimated": None,
            "ground_truth": None,
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
            "Loss and errors come directly from PoseError; "
            "no pose matching is repeated.",
            "Beta is recorded per sample; interpolation settings "
            "are not recorded in PoseError.",
            "If beta_varies is true within a group, its pose-loss "
            "statistics combine different beta weightings.",
            "Overall beta metadata covers successfully analyzed "
            "groups only.",
            "Beta is null when no single value applies; "
            "inspect beta_values and beta_varies.",
            "Unavailable metrics are null, not zero.",
            "Error samples cover published comparisons only, "
            "not every image or input pose.",
            "Accuracy success rate uses recorded valid error "
            "samples as its denominator.",
            "Statistics remain separate for each arena/robot.",
            "Header timestamps identify measurements; "
            "bag timestamps identify receipt times.",
            "Packet telemetry describes producer sessions "
            "and may predate this bag.",
            "Frames-missed percentages assume continuous visibility.",
        ],
    }

    if args.per_sample_output:
        args.per_sample_output.write_text(
            json.dumps(
                all_records, indent=2, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )

    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "bag",
        type=Path,
        help="Bag directory, .db3 file, or .mcap file",
    )
    parser.add_argument(
        "--error-topic",
        default="/pose_error",
        help="Exact PoseError topic or topic prefix",
    )
    parser.add_argument(
        "--arena-frame",
        help="Filter PoseError.header.frame_id",
    )
    parser.add_argument(
        "--bot-frame",
        help="Filter PoseError.bot_label",
    )
    parser.add_argument(
        "--clock",
        choices=("header", "bag"),
        default="header",
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
        "--packet-topic",
        default="/mpose/packets",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
    )
    parser.add_argument(
        "--marker-label",
        help="Select packet telemetry; does not filter errors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Summary JSON file",
    )
    parser.add_argument(
        "--per-sample-output",
        type=Path,
        help="Per-sample JSON grouped by bot and arena",
    )

    args = parser.parse_args(argv)

    for name in (
        "translation_threshold_m",
        "rotation_threshold_deg",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(
                f"--{name.replace('_', '-')} "
                "must be finite and positive"
            )

    if args.no_telemetry and args.marker_label:
        parser.error(
            "--marker-label cannot be used with --no-telemetry"
        )

    return args


def main():
    args = parse_arguments()

    try:
        result = analyze(args)
        text = json.dumps(
            result, indent=2, allow_nan=False
        ) + "\n"

        if args.output:
            args.output.write_text(text, encoding="utf-8")

    except (
        ValueError,
        sqlite3.Error,
        OSError,
        RuntimeError,
        struct.error,
    ) as error:
        raise SystemExit(f"Error: {error}") from error

    print(text, end="")

    if result["successful_arena_bot_groups"] == 0:
        raise SystemExit(
            "Error: no arena/robot group had valid error samples"
        )


if __name__ == "__main__":
    main()