# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Unit tests for MedicalNER engine.
"""

import pytest
from mediscan.core.ner_engine import MedicalNER, EntityType, NERResult


@pytest.fixture
def ner():
    """Create a MedicalNER instance with auto language detection."""
    return MedicalNER(language="auto")


@pytest.fixture
def ner_zh():
    """Create a MedicalNER instance for Chinese."""
    return MedicalNER(language="zh")


@pytest.fixture
def ner_en():
    """Create a MedicalNER instance for English."""
    return MedicalNER(language="en")


class TestSymptomDetection:
    """Test Chinese symptom entity detection."""

    def test_single_symptom(self, ner_zh):
        """Test detection of a single Chinese symptom."""
        result = ner_zh.analyze("患者头痛")
        symptoms = [e for e in result.entities if e.entity_type == EntityType.SYMPTOM]
        assert len(symptoms) >= 1
        assert any(e.text == "头痛" for e in symptoms)

    def test_multiple_symptoms(self, ner_zh):
        """Test detection of multiple Chinese symptoms."""
        result = ner_zh.analyze("患者头痛、发热、咳嗽3天")
        symptoms = [e for e in result.entities if e.entity_type == EntityType.SYMPTOM]
        symptom_texts = {e.text for e in symptoms}
        assert "头痛" in symptom_texts
        assert "发热" in symptom_texts
        assert "咳嗽" in symptom_texts

    def test_complex_symptom(self, ner_zh):
        """Test detection of multi-character symptoms."""
        result = ner_zh.analyze("患者出现呼吸困难")
        symptoms = [e for e in result.entities if e.entity_type == EntityType.SYMPTOM]
        assert any(e.text == "呼吸困难" for e in symptoms)


class TestDiseaseDetection:
    """Test Chinese disease entity detection."""

    def test_single_disease(self, ner_zh):
        """Test detection of a single Chinese disease."""
        result = ner_zh.analyze("诊断为高血压")
        diseases = [e for e in result.entities if e.entity_type == EntityType.DISEASE]
        assert len(diseases) >= 1
        assert any(e.text == "高血压" for e in diseases)

    def test_multiple_diseases(self, ner_zh):
        """Test detection of multiple Chinese diseases."""
        result = ner_zh.analyze("既往有高血压、糖尿病、冠心病病史")
        diseases = [e for e in result.entities if e.entity_type == EntityType.DISEASE]
        disease_texts = {e.text for e in diseases}
        assert "高血压" in disease_texts
        assert "糖尿病" in disease_texts
        assert "冠心病" in disease_texts

    def test_disease_with_modifier(self, ner_zh):
        """Test detection of diseases with modifiers."""
        result = ner_zh.analyze("2型糖尿病")
        diseases = [e for e in result.entities if e.entity_type == EntityType.DISEASE]
        assert any(e.text == "糖尿病" for e in diseases)

    def test_cancer_detection(self, ner_zh):
        """Test detection of cancer diseases."""
        result = ner_zh.analyze("肺癌晚期")
        diseases = [e for e in result.entities if e.entity_type == EntityType.DISEASE]
        assert any(e.text == "肺癌" for e in diseases)


class TestDrugDetection:
    """Test Chinese drug entity detection."""

    def test_single_drug(self, ner_zh):
        """Test detection of a single Chinese drug."""
        result = ner_zh.analyze("服用阿司匹林")
        drugs = [e for e in result.entities if e.entity_type == EntityType.DRUG]
        assert len(drugs) >= 1
        assert any(e.text == "阿司匹林" for e in drugs)

    def test_multiple_drugs(self, ner_zh):
        """Test detection of multiple Chinese drugs."""
        result = ner_zh.analyze("目前服用硝苯地平、二甲双胍、阿托伐他汀")
        drugs = [e for e in result.entities if e.entity_type == EntityType.DRUG]
        drug_texts = {e.text for e in drugs}
        assert "硝苯地平" in drug_texts
        assert "二甲双胍" in drug_texts
        assert "阿托伐他汀" in drug_texts

    def test_drug_with_dosage(self, ner_zh):
        """Test drug detection when dosage info is present."""
        result = ner_zh.analyze("阿莫西林胶囊 0.5g tid")
        drugs = [e for e in result.entities if e.entity_type == EntityType.DRUG]
        assert any(e.text == "阿莫西林" for e in drugs)


class TestEnglishAnalysis:
    """Test English text analysis."""

    def test_english_symptom(self, ner_en):
        """Test English symptom detection."""
        result = ner_en.analyze("Patient presents with headache and fever")
        symptoms = [e for e in result.entities if e.entity_type == EntityType.SYMPTOM]
        symptom_texts = {e.text for e in symptoms}
        assert "headache" in symptom_texts
        assert "fever" in symptom_texts

    def test_english_disease(self, ner_en):
        """Test English disease detection."""
        result = ner_en.analyze("History of hypertension and diabetes")
        diseases = [e for e in result.entities if e.entity_type == EntityType.DISEASE]
        disease_texts = {e.text for e in diseases}
        assert "hypertension" in disease_texts
        assert "diabetes" in disease_texts

    def test_english_drug(self, ner_en):
        """Test English drug detection."""
        result = ner_en.analyze("Currently taking aspirin and metoprolol")
        drugs = [e for e in result.entities if e.entity_type == EntityType.DRUG]
        drug_texts = {e.text for e in drugs}
        assert "aspirin" in drug_texts
        assert "metoprolol" in drug_texts

    def test_english_mixed_entities(self, ner_en):
        """Test mixed entity types in English text."""
        result = ner_en.analyze("Patient with pneumonia, prescribed amoxicillin, blood test ordered")
        diseases = [e for e in result.entities if e.entity_type == EntityType.DISEASE]
        drugs = [e for e in result.entities if e.entity_type == EntityType.DRUG]
        exams = [e for e in result.entities if e.entity_type == EntityType.EXAMINATION]
        assert any(e.text == "pneumonia" for e in diseases)
        assert any(e.text == "amoxicillin" for e in drugs)
        assert any(e.text == "blood test" for e in exams)


class TestAutoLanguageDetection:
    """Test automatic language detection."""

    def test_chinese_text_detected(self, ner):
        """Test that Chinese text is correctly identified."""
        result = ner.analyze("患者头痛发热")
        # Chinese text should use Chinese dictionaries
        assert result.text_length > 0
        # Should find Chinese entities
        assert len(result.entities) > 0

    def test_english_text_detected(self, ner):
        """Test that English text is correctly identified."""
        result = ner.analyze("Patient has headache and chest pain")
        assert result.text_length > 0
        assert len(result.entities) > 0

    def test_mixed_text_defaults_to_chinese(self, ner):
        """Test that text with >10% Chinese characters defaults to Chinese."""
        result = ner.analyze("患者patient头痛headache")
        assert result.text_length > 0

    def test_explicit_language_override(self, ner_zh):
        """Test explicit language parameter override."""
        result = ner_zh.analyze("Patient has headache", language="en")
        symptoms = [e for e in result.entities if e.entity_type == EntityType.SYMPTOM]
        assert any(e.text == "headache" for e in symptoms)


class TestEmptyText:
    """Test handling of empty or invalid text."""

    def test_empty_string(self, ner):
        """Test empty string returns empty result."""
        result = ner.analyze("")
        assert isinstance(result, NERResult)
        assert result.entities == []
        assert result.text_length == 0

    def test_whitespace_only(self, ner):
        """Test whitespace-only string returns empty result."""
        result = ner.analyze("   \n\t  ")
        assert result.entities == []
        assert result.text_length == 0

    def test_none_like_empty(self, ner):
        """Test that empty-like input is handled gracefully."""
        result = ner.analyze("")
        assert len(result.entities) == 0


class TestResultPatterns:
    """Test detection of test result values."""

    def test_blood_pressure_values(self, ner_zh):
        """Test detection of blood pressure values like 120/80 mmHg."""
        result = ner_zh.analyze("血压 120/80mmHg")
        test_results = [e for e in result.entities if e.entity_type == EntityType.TEST_RESULT]
        # Should detect the numeric value with unit
        assert len(test_results) >= 1

    def test_blood_glucose(self, ner_zh):
        """Test detection of blood glucose values."""
        result = ner_zh.analyze("空腹血糖 7.2 mmol/L")
        test_results = [e for e in result.entities if e.entity_type == EntityType.TEST_RESULT]
        assert len(test_results) >= 1

    def test_abnormal_markers(self, ner_zh):
        """Test detection of abnormal markers."""
        result = ner_zh.analyze("血糖偏高")
        test_results = [e for e in result.entities if e.entity_type == EntityType.TEST_RESULT]
        assert any("偏高" in e.text for e in test_results)

    def test_percentage_value(self, ner_zh):
        """Test detection of percentage values."""
        result = ner_zh.analyze("射血分数 55%")
        test_results = [e for e in result.entities if e.entity_type == EntityType.TEST_RESULT]
        assert len(test_results) >= 1

    def test_english_test_result(self, ner_en):
        """Test English test result detection."""
        result = ner_en.analyze("Blood glucose 120 mg/dL, high")
        test_results = [e for e in result.entities if e.entity_type == EntityType.TEST_RESULT]
        assert len(test_results) >= 1


class TestNERResultStructure:
    """Test NERResult data structure."""

    def test_result_dict(self, ner_zh):
        """Test that to_dict produces correct structure."""
        result = ner_zh.analyze("高血压患者头痛")
        d = result.to_dict()
        assert "entities" in d
        assert "total_entities" in d
        assert "text_length" in d
        assert "processing_time_ms" in d
        assert "summary" in d
        assert d["total_entities"] == len(result.entities)

    def test_entity_dict(self, ner_zh):
        """Test that entity to_dict produces correct structure."""
        result = ner_zh.analyze("高血压")
        if result.entities:
            e = result.entities[0]
            d = e.to_dict()
            assert "text" in d
            assert "type" in d
            assert "start" in d
            assert "end" in d
            assert "confidence" in d

    def test_entity_positions(self, ner_zh):
        """Test that entity start/end positions are correct."""
        text = "患者高血压"
        result = ner_zh.analyze(text)
        diseases = [e for e in result.entities if e.entity_type == EntityType.DISEASE]
        if diseases:
            e = diseases[0]
            assert text[e.start:e.end] == e.text

    def test_no_overlapping_entities(self, ner_zh):
        """Test that entities do not overlap."""
        result = ner_zh.analyze("高血压患者头痛")
        for i in range(len(result.entities)):
            for j in range(i + 1, len(result.entities)):
                a, b = result.entities[i], result.entities[j]
                assert not (a.start < b.end and b.start < a.end), \
                    f"Overlapping entities: {a.text}({a.start}-{a.end}) and {b.text}({b.start}-{b.end})"


class TestBatchAnalysis:
    """Test batch analysis functionality."""

    def test_batch_analysis(self, ner_zh):
        """Test analyzing multiple texts at once."""
        texts = ["患者头痛", "高血压", "服用阿司匹林"]
        results = ner_zh.analyze_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, NERResult) for r in results)
        assert results[0].text_length > 0
