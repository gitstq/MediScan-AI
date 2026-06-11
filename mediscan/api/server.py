# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Flask REST API for MediScan-AI.
Provides endpoints for NER, PII masking, record parsing, and drug interaction checking.
"""

import json
from flask import Flask, request, jsonify, send_from_directory
from mediscan.core.ner_engine import MedicalNER
from mediscan.core.pii_masker import PIIMasker, MaskStrategy
from mediscan.core.record_parser import RecordParser
from mediscan.core.drug_checker import DrugChecker


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        static_folder="web/static",
        template_folder="web/templates",
    )

    # Initialize engines
    ner_engine = MedicalNER(language="auto")
    pii_masker = PIIMasker(strategy=MaskStrategy.REDACT)
    record_parser = RecordParser()
    drug_checker = DrugChecker()

    # --- API Routes ---

    @app.route("/")
    def index():
        """Serve the main web interface."""
        return send_from_directory(app.template_folder, "index.html")

    @app.route("/api/health")
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "engine": "MediScan-AI",
        })

    @app.route("/api/ner", methods=["POST"])
    def analyze_ner():
        """
        Medical Named Entity Recognition.
        Request body: {"text": "...", "language": "zh|en|auto"}
        """
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        language = data.get("language", "auto")

        result = ner_engine.analyze(text, language=language)
        return jsonify(result.to_dict())

    @app.route("/api/pii/mask", methods=["POST"])
    def mask_pii():
        """
        PII Detection and Masking.
        Request body: {"text": "...", "strategy": "redact|hash|partial|tag", "exclude": ["name_zh"]}
        """
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        strategy_name = data.get("strategy", "redact")
        exclude_types = data.get("exclude", None)

        strategy_map = {
            "redact": MaskStrategy.REDACT,
            "hash": MaskStrategy.HASH,
            "partial": MaskStrategy.PARTIAL,
            "tag": MaskStrategy.TAG,
            "replace": MaskStrategy.REPLACE,
        }
        strategy = strategy_map.get(strategy_name, MaskStrategy.REDACT)

        result = pii_masker.mask(text, strategy=strategy, exclude_types=exclude_types)
        return jsonify(result.to_dict())

    @app.route("/api/pii/detect", methods=["POST"])
    def detect_pii():
        """
        PII Detection (dry run, no masking).
        Request body: {"text": "..."}
        """
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        pii_list = pii_masker.detect(text)
        return jsonify({
            "pii_found": [p.to_dict() for p in pii_list],
            "total_pii": len(pii_list),
        })

    @app.route("/api/parse", methods=["POST"])
    def parse_record():
        """
        Clinical Record Parsing.
        Request body: {"text": "..."}
        """
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        result = record_parser.parse(text)
        return jsonify(result.to_dict())

    @app.route("/api/drug/check", methods=["POST"])
    def check_drugs():
        """
        Drug Interaction Check.
        Request body: {"drugs": ["drug1", "drug2", ...]}
        """
        data = request.get_json(silent=True)
        if not data or "drugs" not in data:
            return jsonify({"error": "Missing 'drugs' field"}), 400

        drugs = data["drugs"]
        if not isinstance(drugs, list):
            return jsonify({"error": "'drugs' must be a list"}), 400

        result = drug_checker.check(drugs)
        return jsonify(result.to_dict())

    @app.route("/api/analyze", methods=["POST"])
    def full_analysis():
        """
        Full analysis pipeline: NER + PII detection + Record parsing.
        Request body: {"text": "...", "language": "zh|en|auto", "mask_pii": true}
        """
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        language = data.get("language", "auto")
        mask_pii = data.get("mask_pii", True)

        # Run all analyses
        ner_result = ner_engine.analyze(text, language=language)
        pii_result = pii_masker.mask(text) if mask_pii else pii_masker.detect(text)
        record_result = record_parser.parse(text)

        return jsonify({
            "ner": ner_result.to_dict(),
            "pii": pii_result.to_dict() if mask_pii else {
                "pii_found": [p.to_dict() for p in pii_result],
                "total_pii": len(pii_result),
            },
            "record": record_result.to_dict(),
        })

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
