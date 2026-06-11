# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Unit tests for RecordParser.
"""

import pytest
from mediscan.core.record_parser import RecordParser, RecordType, ParsedRecord, ParsedField


@pytest.fixture
def parser():
    """Create a RecordParser instance."""
    return RecordParser()


SAMPLE_OUTPATIENT = """门诊记录
患者：李四，男，65岁
主诉：反复头痛、头晕2年
现病史：患者2年前无明显诱因出现头痛、头晕，以枕部为主，伴有恶心。
既往史：高血压病10年，糖尿病5年
过敏史：青霉素过敏
体格检查：血压160/100mmHg，心率80次/分
辅助检查：头颅CT未见异常
诊断：高血压病3级，2型糖尿病
处理：硝苯地平控释片 30mg qd，二甲双胍 0.5g tid
医师：张医生
"""

SAMPLE_INPATIENT = """住院记录
患者：王五，女，72岁
入院诊断：慢性心力衰竭，心房颤动
主诉：活动后气短1月，加重3天
既往史：高血压病20年，冠心病10年
"""

SAMPLE_PRESCRIPTION = """处方
患者赵六，男，45岁
Rp
阿司匹林肠溶片 100mg x30
用法：100mg qd po
阿托伐他汀钙片 20mg x30
用法：20mg qn po
"""

SAMPLE_LAB_REPORT = """检验报告
患者：钱七
血常规：WBC 6.5×10⁹/L，RBC 4.2×10¹²/L，Hb 130g/L
肝功能：ALT 25U/L，AST 20U/L
肾功能：Cr 80μmol/L，BUN 5.2mmol/L
"""


class TestOutpatientRecord:
    """Test outpatient record parsing."""

    def test_outpatient_type_detection(self, parser):
        """Test detection of outpatient record type."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        assert result.record_type == RecordType.OUTPATIENT

    def test_outpatient_field_extraction(self, parser):
        """Test extraction of fields from outpatient record."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        field_names = {f.field_name_en for f in result.fields}
        assert "chief_complaint" in field_names
        assert "present_illness" in field_names
        assert "past_history" in field_names
        assert "allergy_history" in field_names
        assert "physical_exam" in field_names
        assert "auxiliary_exam" in field_names
        assert "diagnosis" in field_names
        assert "treatment_plan" in field_names
        assert "doctor" in field_names

    def test_chief_complaint_content(self, parser):
        """Test chief complaint field content."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        cc = [f for f in result.fields if f.field_name_en == "chief_complaint"]
        assert len(cc) == 1
        assert "头痛" in cc[0].value or "头晕" in cc[0].value

    def test_diagnosis_content(self, parser):
        """Test diagnosis field content."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        diag = [f for f in result.fields if f.field_name_en == "diagnosis"]
        assert len(diag) == 1
        assert "高血压" in diag[0].value

    def test_raw_text_preserved(self, parser):
        """Test that raw text is preserved in result."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        assert result.raw_text == SAMPLE_OUTPATIENT


class TestInpatientRecord:
    """Test inpatient record detection."""

    def test_inpatient_type_detection(self, parser):
        """Test detection of inpatient record type."""
        result = parser.parse(SAMPLE_INPATIENT)
        assert result.record_type == RecordType.INPATIENT

    def test_inpatient_fields(self, parser):
        """Test field extraction from inpatient record."""
        result = parser.parse(SAMPLE_INPATIENT)
        field_names = {f.field_name_en for f in result.fields}
        assert "chief_complaint" in field_names
        assert "past_history" in field_names
        assert "diagnosis" in field_names


class TestPrescriptionDetection:
    """Test prescription record detection."""

    def test_prescription_type_detection(self, parser):
        """Test detection of prescription record type."""
        result = parser.parse(SAMPLE_PRESCRIPTION)
        assert result.record_type == RecordType.PRESCRIPTION

    def test_prescription_fields(self, parser):
        """Test field extraction from prescription."""
        result = parser.parse(SAMPLE_PRESCRIPTION)
        field_names = {f.field_name_en for f in result.fields}
        assert "prescription" in field_names or "patient_info" in field_names


class TestLabReport:
    """Test lab report detection."""

    def test_lab_report_type_detection(self, parser):
        """Test detection of lab report type."""
        result = parser.parse(SAMPLE_LAB_REPORT)
        assert result.record_type == RecordType.LAB_REPORT


class TestFieldExtraction:
    """Test specific field extraction."""

    def test_patient_info_extraction(self, parser):
        """Test patient info extraction."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        patient = [f for f in result.fields if f.field_name_en == "patient_info"]
        assert len(patient) >= 1
        assert "李四" in patient[0].value

    def test_allergy_history_extraction(self, parser):
        """Test allergy history extraction."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        allergy = [f for f in result.fields if f.field_name_en == "allergy_history"]
        assert len(allergy) == 1
        assert "青霉素" in allergy[0].value

    def test_treatment_plan_extraction(self, parser):
        """Test treatment plan extraction."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        treatment = [f for f in result.fields if f.field_name_en == "treatment_plan"]
        assert len(treatment) == 1
        assert "硝苯地平" in treatment[0].value or "二甲双胍" in treatment[0].value

    def test_doctor_extraction(self, parser):
        """Test doctor name extraction."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        doctor = [f for f in result.fields if f.field_name_en == "doctor"]
        assert len(doctor) == 1
        assert "张" in doctor[0].value


class TestUnknownRecordType:
    """Test unknown record type handling."""

    def test_unknown_type(self, parser):
        """Test that unrecognizable text returns UNKNOWN type."""
        text = "这是一段普通文本，没有任何病历特征。"
        result = parser.parse(text)
        assert result.record_type == RecordType.UNKNOWN

    def test_unknown_still_extracts_fields(self, parser):
        """Test that unknown type still attempts field extraction."""
        text = "患者：测试，诊断：感冒"
        result = parser.parse(text)
        assert result.record_type == RecordType.UNKNOWN
        # Should still extract some fields
        assert len(result.fields) >= 1


class TestEmptyText:
    """Test handling of empty text."""

    def test_empty_string(self, parser):
        """Test empty string returns UNKNOWN type."""
        result = parser.parse("")
        assert result.record_type == RecordType.UNKNOWN
        assert result.fields == []

    def test_whitespace_only(self, parser):
        """Test whitespace-only string."""
        result = parser.parse("   \n\t  ")
        assert result.record_type == RecordType.UNKNOWN

    def test_empty_result_structure(self, parser):
        """Test that empty result has correct structure."""
        result = parser.parse("")
        assert isinstance(result, ParsedRecord)
        assert result.raw_text == ""


class TestParsedResultStructure:
    """Test ParsedRecord data structure."""

    def test_result_dict(self, parser):
        """Test that to_dict produces correct structure."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        d = result.to_dict()
        assert "record_type" in d
        assert "fields" in d
        assert "summary" in d
        assert d["record_type"] == "outpatient"

    def test_field_dict(self, parser):
        """Test that ParsedField to_dict produces correct structure."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        if result.fields:
            f = result.fields[0]
            d = f.to_dict()
            assert "field" in d
            assert "field_en" in d
            assert "value" in d
            assert "confidence" in d

    def test_summary_dict(self, parser):
        """Test summary dict in result."""
        result = parser.parse(SAMPLE_OUTPATIENT)
        d = result.to_dict()
        summary = d["summary"]
        assert isinstance(summary, dict)
        # Summary should map field_en to value
        if result.fields:
            first_field = result.fields[0]
            assert first_field.field_name_en in summary


class TestBatchParsing:
    """Test batch parsing functionality."""

    def test_batch_parse(self, parser):
        """Test parsing multiple records."""
        texts = [SAMPLE_OUTPATIENT, SAMPLE_INPATIENT, SAMPLE_PRESCRIPTION]
        results = parser.parse_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, ParsedRecord) for r in results)
        assert results[0].record_type == RecordType.OUTPATIENT
        assert results[1].record_type == RecordType.INPATIENT
        assert results[2].record_type == RecordType.PRESCRIPTION
