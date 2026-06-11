# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Unit tests for DrugChecker.
"""

import pytest
from mediscan.core.drug_checker import DrugChecker, InteractionSeverity, CheckResult


@pytest.fixture
def checker():
    """Create a DrugChecker instance."""
    return DrugChecker()


class TestCriticalInteraction:
    """Test detection of critical drug interactions."""

    def test_aspirin_warfarin(self, checker):
        """Test aspirin + warfarin critical interaction."""
        result = checker.check(["阿司匹林", "华法林"])
        assert result.total_pairs == 1
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.CRITICAL
        assert "出血" in result.interactions[0].description

    def test_metronidazole_warfarin(self, checker):
        """Test metronidazole + warfarin critical interaction."""
        result = checker.check(["甲硝唑", "华法林"])
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.CRITICAL


class TestMajorInteraction:
    """Test detection of major drug interactions."""

    def test_clopidogrel_omeprazole(self, checker):
        """Test clopidogrel + omeprazole major interaction."""
        result = checker.check(["氯吡格雷", "奥美拉唑"])
        assert result.total_pairs == 1
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.MAJOR
        assert "CYP2C19" in result.interactions[0].mechanism

    def test_simvastatin_amlodipine(self, checker):
        """Test simvastatin + amlodipine major interaction."""
        result = checker.check(["辛伐他汀", "氨氯地平"])
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.MAJOR

    def test_ibuprofen_warfarin(self, checker):
        """Test ibuprofen + warfarin major interaction."""
        result = checker.check(["布洛芬", "华法林"])
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.MAJOR


class TestSafeDrugPair:
    """Test safe drug pairs (no known interaction)."""

    def test_no_interaction(self, checker):
        """Test drugs with no known interaction."""
        result = checker.check(["阿司匹林", "二甲双胍"])
        assert result.total_pairs == 1
        assert len(result.interactions) == 0
        assert result.safe_pairs == 1

    def test_safe_pair_result(self, checker):
        """Test that safe pair result has correct structure."""
        result = checker.check(["氯雷他定", "西替利嗪"])
        assert result.total_pairs == 1
        assert result.safe_pairs == 1
        assert len(result.interactions) == 0

    def test_mixed_safe_and_interacting(self, checker):
        """Test mixed set with both safe and interacting pairs."""
        result = checker.check(["阿司匹林", "华法林", "二甲双胍"])
        # aspirin-warfarin: critical, aspirin-metformin: safe, warfarin-metformin: safe
        assert result.total_pairs == 3
        assert len(result.interactions) == 1
        assert result.safe_pairs == 2
        assert result.interactions[0].severity == InteractionSeverity.CRITICAL


class TestDrugAliases:
    """Test drug name alias resolution."""

    def test_english_aspirin(self, checker):
        """Test English name 'aspirin' maps to Chinese canonical name."""
        result = checker.check(["aspirin", "warfarin"])
        assert result.total_pairs == 1
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.CRITICAL

    def test_brand_name_fenbid(self, checker):
        """Test brand name '芬必得' maps to ibuprofen."""
        result = checker.check(["芬必得", "华法林"])
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.MAJOR

    def test_brand_name_bolewei(self, checker):
        """Test brand name '波立维' maps to clopidogrel."""
        result = checker.check(["波立维", "奥美拉唑"])
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.MAJOR

    def test_mixed_chinese_english(self, checker):
        """Test mixed Chinese and English drug names."""
        result = checker.check(["aspirin", "华法林"])
        assert len(result.interactions) == 1
        assert result.interactions[0].severity == InteractionSeverity.CRITICAL


class TestEmptyDrugList:
    """Test handling of empty drug list."""

    def test_empty_list(self, checker):
        """Test empty drug list returns empty result."""
        result = checker.check([])
        assert isinstance(result, CheckResult)
        assert result.drugs_checked == []
        assert result.total_pairs == 0
        assert result.interactions == []
        assert result.safe_pairs == 0

    def test_single_drug(self, checker):
        """Test single drug returns no pairs."""
        result = checker.check(["阿司匹林"])
        assert result.total_pairs == 0
        assert len(result.interactions) == 0

    def test_unknown_drug(self, checker):
        """Test completely unknown drug is ignored."""
        result = checker.check(["未知药物A", "未知药物B"])
        # Unknown drugs won't normalize, so no pairs
        assert result.total_pairs == 0

    def test_partial_unknown(self, checker):
        """Test mix of known and unknown drugs."""
        result = checker.check(["阿司匹林", "不存在的药"])
        # Only aspirin normalizes, single drug = no pairs
        assert result.total_pairs == 0


class TestCheckResultStructure:
    """Test CheckResult data structure."""

    def test_result_dict(self, checker):
        """Test that to_dict produces correct structure."""
        result = checker.check(["阿司匹林", "华法林"])
        d = result.to_dict()
        assert "drugs_checked" in d
        assert "total_pairs_checked" in d
        assert "interactions_found" in d
        assert "safe_pairs" in d
        assert "severity_summary" in d
        assert "interactions" in d
        assert "has_critical" in d
        assert "has_major" in d
        assert d["has_critical"] is True

    def test_interaction_dict(self, checker):
        """Test that DrugInteraction to_dict produces correct structure."""
        result = checker.check(["阿司匹林", "华法林"])
        if result.interactions:
            i = result.interactions[0]
            d = i.to_dict()
            assert "drug_a" in d
            assert "drug_b" in d
            assert "severity" in d
            assert "description" in d
            assert "mechanism" in d
            assert "recommendation" in d

    def test_severity_summary(self, checker):
        """Test severity summary aggregation."""
        result = checker.check(["阿司匹林", "华法林", "布洛芬"])
        d = result.to_dict()
        assert "critical" in d["severity_summary"]
        # aspirin-warfarin: critical, aspirin-ibuprofen: moderate, warfarin-ibuprofen: major
        assert d["severity_summary"]["critical"] == 1
        assert d["severity_summary"]["major"] == 1
        assert d["severity_summary"]["moderate"] == 1


class TestDuplicateDrugs:
    """Test handling of duplicate drug entries."""

    def test_duplicate_drug_names(self, checker):
        """Test that duplicate drug names are deduplicated."""
        result = checker.check(["阿司匹林", "阿司匹林", "华法林"])
        assert len(result.drugs_checked) == 2
        assert result.total_pairs == 1

    def test_alias_duplicate(self, checker):
        """Test that drug aliases resolve to same canonical name."""
        result = checker.check(["aspirin", "阿司匹林", "华法林"])
        # aspirin and 阿司匹林 should resolve to same canonical name
        assert len(result.drugs_checked) == 2
        assert result.total_pairs == 1
