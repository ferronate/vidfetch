#!/usr/bin/env python3
"""
Correction review CLI for vidfetch.
Usage: python -m scripts.review_corrections [options]

Examples:
    python -m scripts.review_corrections                    # Show summary
    python -m scripts.review_corrections --list             # List all corrections
    python -m scripts.review_corrections --rules            # List all rules
    python -m scripts.review_corrections --add-rule dog cat # Add a rule
    python -m scripts.review_corrections --stats            # Show class statistics
    python -m scripts.review_corrections --export           # Export to JSON
"""
import argparse
import sys
from pathlib import Path

from src.corrections import Correction, CorrectionRule, get_store
from src.correction_applier import get_correction_summary


def cmd_summary(store):
    """Show correction system summary."""
    summary = get_correction_summary()
    
    print("\n" + "=" * 60)
    print("CORRECTION SYSTEM SUMMARY")
    print("=" * 60)
    print(f"Total corrections:  {summary['total_corrections']}")
    print(f"Total rules:        {summary['total_rules']}")
    print(f"Active rules:       {summary['active_rules']}")
    
    if summary['class_stats']:
        print(f"\nClass Statistics:")
        print(f"  {'Class':<20} {'Detections':>10} {'Corrections':>12} {'Rate':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8}")
        for cls, stats in summary['class_stats'].items():
            print(f"  {cls:<20} {stats['total_detections']:>10} {stats['total_corrections']:>12} {stats['correction_rate']:>7.1%}")
    
    if summary['recent_corrections']:
        print(f"\nRecent Corrections:")
        for c in summary['recent_corrections'][:10]:
            print(f"  [{c['video']}] {c['original']} → {c['corrected']} ({c['action']})")
    
    print("=" * 60)


def cmd_list_corrections(store, args):
    """List corrections."""
    corrections = store.get_corrections(
        video_id=args.video,
        original_class=args.class_name,
        limit=args.limit,
    )
    
    if not corrections:
        print("No corrections found.")
        return
    
    print(f"\n{'ID':<5} {'Video':<25} {'Original':<15} {'Corrected':<15} {'Conf':>6} {'Action':<10}")
    print("-" * 80)
    
    for c in corrections:
        print(f"{c.id:<5} {c.video_id:<25} {c.original_class:<15} {c.corrected_class:<15} {c.original_confidence:>6.3f} {c.action:<10}")
    
    print(f"\nTotal: {len(corrections)} corrections")


def cmd_list_rules(store, args):
    """List correction rules."""
    rules = store.get_rules(enabled_only=not args.all)
    
    if not rules:
        print("No rules found.")
        return
    
    print(f"\n{'ID':<5} {'Pattern':<15} {'Target':<15} {'Conf <':>7} {'Video':<15} {'Uses':>5} {'Status':<8}")
    print("-" * 75)
    
    for r in rules:
        conf_str = f"{r.confidence_threshold:.2f}" if r.confidence_threshold else "any"
        video_str = r.video_pattern or "*"
        status = "ON" if r.enabled else "OFF"
        print(f"{r.id:<5} {r.pattern_class:<15} {r.target_class:<15} {conf_str:>7} {video_str:<15} {r.usage_count:>5} {status:<8}")
    
    print(f"\nTotal: {len(rules)} rules")


def cmd_add_rule(store, args):
    """Add a correction rule."""
    rule = CorrectionRule(
        pattern_class=args.pattern,
        target_class=args.target,
        confidence_threshold=args.confidence,
        video_pattern=args.video_pattern,
    )
    
    rule_id = store.add_rule(rule)
    print(f"✓ Rule added (ID: {rule_id}): {args.pattern} → {args.target}")
    
    if args.confidence:
        print(f"  Applies when confidence < {args.confidence}")
    if args.video_pattern:
        print(f"  Video pattern: {args.video_pattern}")


def cmd_toggle_rule(store, args):
    """Toggle a rule on/off."""
    store.toggle_rule(args.rule_id, not args.disable)
    status = "disabled" if args.disable else "enabled"
    print(f"✓ Rule #{args.rule_id} {status}")


def cmd_delete_rule(store, args):
    """Delete a rule."""
    store.delete_rule(args.rule_id)
    print(f"✓ Rule #{args.rule_id} deleted")


def cmd_delete_correction(store, args):
    """Delete a correction."""
    store.delete_correction(args.correction_id)
    print(f"✓ Correction #{args.correction_id} deleted")


def cmd_generate_rules(store, args):
    """Auto-generate rules from corrections."""
    count = store.generate_rules_from_corrections(min_occurrences=args.min)
    print(f"✓ Generated {count} rule(s) from repeated corrections")


def cmd_stats(store, args):
    """Show detailed class statistics."""
    stats = store.get_class_stats()
    
    if not stats:
        print("No class statistics available.")
        return
    
    print(f"\n{'Class':<20} {'Detections':>10} {'Corrections':>12} {'Avg Conf':>10} {'Correction Rate':>16}")
    print("-" * 75)
    
    for cls, s in sorted(stats.items(), key=lambda x: x[1]['total_detections'], reverse=True):
        print(f"{cls:<20} {s['total_detections']:>10} {s['total_corrections']:>12} {s['avg_confidence']:>10.3f} {s['correction_rate']:>15.1%}")


def cmd_export(store, args):
    """Export corrections to JSON."""
    output = Path(args.output) if args.output else Path("data/corrections_export.json")
    path = store.export_corrections(output)
    print(f"✓ Exported to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Review and manage object detection corrections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Main actions
    parser.add_argument("--list", "-l", action="store_true", help="List all corrections")
    parser.add_argument("--rules", "-r", action="store_true", help="List all rules")
    parser.add_argument("--stats", "-s", action="store_true", help="Show class statistics")
    parser.add_argument("--export", "-e", action="store_true", help="Export corrections to JSON")
    
    # Filters
    parser.add_argument("--video", "-v", type=str, help="Filter by video ID")
    parser.add_argument("--class", "-c", dest="class_name", type=str, help="Filter by class name")
    parser.add_argument("--limit", type=int, default=50, help="Max corrections to show (default: 50)")
    parser.add_argument("--all", action="store_true", help="Show all (including disabled rules)")
    
    # Rule management
    parser.add_argument("--add-rule", nargs=2, metavar=("PATTERN", "TARGET"),
                        help="Add a rule: --add-rule dog cat")
    parser.add_argument("--rule-conf", type=float, help="Confidence threshold for new rule")
    parser.add_argument("--rule-video", type=str, help="Video pattern for new rule")
    parser.add_argument("--toggle", type=int, metavar="RULE_ID", help="Toggle rule on/off")
    parser.add_argument("--disable", action="store_true", help="Disable (used with --toggle)")
    parser.add_argument("--delete-rule", type=int, metavar="RULE_ID", help="Delete a rule")
    parser.add_argument("--delete-correction", type=int, metavar="CORRECTION_ID", help="Delete a correction")
    parser.add_argument("--generate-rules", action="store_true", help="Auto-generate rules from corrections")
    parser.add_argument("--min-occurrences", type=int, default=3, help="Min occurrences for auto-rule (default: 3)")
    
    # Export
    parser.add_argument("--output", "-o", type=str, help="Output file for export")
    
    args = parser.parse_args()
    store = get_store()
    
    # Default: show summary
    if not any([args.list, args.rules, args.stats, args.export,
                args.add_rule, args.toggle, args.delete_rule,
                args.delete_correction, args.generate_rules]):
        cmd_summary(store)
        return
    
    # Execute requested action
    if args.list:
        cmd_list_corrections(store, args)
    elif args.rules:
        cmd_list_rules(store, args)
    elif args.stats:
        cmd_stats(store, args)
    elif args.export:
        cmd_export(store, args)
    elif args.add_rule:
        args.pattern, args.target = args.add_rule
        args.confidence = args.rule_conf
        args.video_pattern = args.rule_video
        cmd_add_rule(store, args)
    elif args.toggle is not None:
        args.rule_id = args.toggle
        cmd_toggle_rule(store, args)
    elif args.delete_rule is not None:
        args.rule_id = args.delete_rule
        cmd_delete_rule(store, args)
    elif args.delete_correction is not None:
        args.correction_id = args.delete_correction
        cmd_delete_correction(store, args)
    elif args.generate_rules:
        args.min = args.min_occurrences
        cmd_generate_rules(store, args)


if __name__ == "__main__":
    main()
