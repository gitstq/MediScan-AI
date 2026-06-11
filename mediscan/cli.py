# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Command Line Interface for MediScan-AI.
Provides commands for NER, PII masking, record parsing, and drug interaction checking.
"""

import sys
import json
import argparse
from mediscan.core.ner_engine import MedicalNER
from mediscan.core.pii_masker import PIIMasker, MaskStrategy
from mediscan.core.record_parser import RecordParser
from mediscan.core.drug_checker import DrugChecker


def cmd_ner(args):
    """Run NER analysis."""
    ner = MedicalNER(language=args.language)
    result = ner.analyze(args.text, language=args.language)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def cmd_pii(args):
    """Run PII masking."""
    masker = PIIMasker(strategy=MaskStrategy(args.strategy))
    result = masker.mask(args.text, strategy=MaskStrategy(args.strategy))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def cmd_parse(args):
    """Parse clinical record."""
    parser = RecordParser()
    result = parser.parse(args.text)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def cmd_drug(args):
    """Check drug interactions."""
    checker = DrugChecker()
    result = checker.check(args.drugs)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def cmd_serve(args):
    """Start web server."""
    from mediscan.api.server import create_app
    app = create_app()
    print(f"MediScan-AI Web Server starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mediscan",
        description="MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # NER command
    ner_parser = subparsers.add_parser("ner", help="Medical Named Entity Recognition")
    ner_parser.add_argument("text", help="Clinical text to analyze")
    ner_parser.add_argument("--language", "-l", default="auto", choices=["zh", "en", "auto"],
                             help="Language (default: auto-detect)")

    # PII command
    pii_parser = subparsers.add_parser("pii", help="PII Detection and Masking")
    pii_parser.add_argument("text", help="Text containing PII")
    pii_parser.add_argument("--strategy", "-s", default="redact",
                            choices=["redact", "hash", "partial", "tag", "replace"],
                            help="Masking strategy (default: redact)")

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Clinical Record Parsing")
    parse_parser.add_argument("text", help="Raw clinical record text")

    # Drug command
    drug_parser = subparsers.add_parser("drug", help="Drug Interaction Check")
    drug_parser.add_argument("drugs", nargs="+", help="Drug names to check")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start Web Server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", "-p", type=int, default=5000, help="Port (default: 5000)")
    serve_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.command == "ner":
        cmd_ner(args)
    elif args.command == "pii":
        cmd_pii(args)
    elif args.command == "parse":
        cmd_parse(args)
    elif args.command == "drug":
        cmd_drug(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
