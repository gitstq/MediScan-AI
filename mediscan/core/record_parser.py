# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Clinical Record Parser.
Parses unstructured Chinese clinical records into structured format.
Supports outpatient records, inpatient records, prescriptions, and lab reports.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class RecordType(Enum):
    """Clinical record types."""
    OUTPATIENT = "outpatient"        # 门诊记录
    INPATIENT = "inpatient"          # 住院记录
    PRESCRIPTION = "prescription"    # 处方
    LAB_REPORT = "lab_report"        # 检验报告
    DISCHARGE = "discharge"          # 出院小结
    UNKNOWN = "unknown"


@dataclass
class ParsedField:
    """A single parsed field from a clinical record."""
    field_name: str
    field_name_en: str
    value: str
    confidence: float = 0.8

    def to_dict(self) -> Dict:
        return {
            "field": self.field_name,
            "field_en": self.field_name_en,
            "value": self.value,
            "confidence": self.confidence,
        }


@dataclass
class ParsedRecord:
    """Complete parsed clinical record."""
    record_type: RecordType
    fields: List[ParsedField] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict:
        return {
            "record_type": self.record_type.value,
            "fields": [f.to_dict() for f in self.fields],
            "summary": {f.field_name_en: f.value for f in self.fields},
        }


class RecordParser:
    """
    Clinical Record Structure Parser.
    Extracts structured fields from unstructured Chinese clinical text.
    Uses pattern matching and section detection.
    """

    # Section patterns for Chinese clinical records
    SECTION_PATTERNS = {
        "chief_complaint": {
            "zh": ["主诉", "主 诉"],
            "en": "chief_complaint",
        },
        "present_illness": {
            "zh": ["现病史", "现 病 史"],
            "en": "present_illness",
        },
        "past_history": {
            "zh": ["既往史", "既 往 史"],
            "en": "past_history",
        },
        "allergy_history": {
            "zh": ["过敏史", "过 敏 史", "药物过敏"],
            "en": "allergy_history",
        },
        "family_history": {
            "zh": ["家族史", "家 族 史"],
            "en": "family_history",
        },
        "physical_exam": {
            "zh": ["体格检查", "体 格 检 查", "查体"],
            "en": "physical_exam",
        },
        "auxiliary_exam": {
            "zh": ["辅助检查", "辅 助 检 查"],
            "en": "auxiliary_exam",
        },
        "diagnosis": {
            "zh": ["诊断", "诊 断", "初步诊断", "最终诊断", "入院诊断", "出院诊断"],
            "en": "diagnosis",
        },
        "treatment_plan": {
            "zh": ["处理", "治疗方案", "治 疗 方 案", "治疗意见", "医嘱"],
            "en": "treatment_plan",
        },
        "prescription": {
            "zh": ["处方", "处 方", "用药"],
            "en": "prescription",
        },
        "doctor": {
            "zh": ["医师", "主治医师", "住院医师", "主任医师", "副主任医师", "签名"],
            "en": "doctor",
        },
        "department": {
            "zh": ["科室", "科别", "科 室"],
            "en": "department",
        },
        "date": {
            "zh": ["日期", "时间", "记录时间", "就诊时间"],
            "en": "date",
        },
        "patient_info": {
            "zh": ["患者", "姓名", "性别", "年龄", "床号"],
            "en": "patient_info",
        },
    }

    # Record type detection keywords
    TYPE_KEYWORDS = {
        RecordType.OUTPATIENT: ["门诊", "门诊记录", "就诊"],
        RecordType.INPATIENT: ["住院", "住院记录", "入院记录"],
        RecordType.PRESCRIPTION: ["处方", "处方笺", "Rp"],
        RecordType.LAB_REPORT: ["检验报告", "化验单", "检查结果", "报告单"],
        RecordType.DISCHARGE: ["出院小结", "出院记录", "出院总结"],
    }

    def parse(self, text: str) -> ParsedRecord:
        """
        Parse unstructured clinical text into structured format.

        Args:
            text: Raw clinical record text.

        Returns:
            ParsedRecord with extracted fields.
        """
        if not text or not text.strip():
            return ParsedRecord(record_type=RecordType.UNKNOWN)

        record_type = self._detect_type(text)
        fields = self._extract_fields(text)

        return ParsedRecord(
            record_type=record_type,
            fields=fields,
            raw_text=text,
        )

    def _detect_type(self, text: str) -> RecordType:
        """Detect the type of clinical record."""
        for rtype, keywords in self.TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return rtype
        return RecordType.UNKNOWN

    def _extract_fields(self, text: str) -> List[ParsedField]:
        """Extract structured fields from clinical text."""
        fields = []

        # Extract patient info (name, gender, age, etc.)
        patient_info = self._extract_patient_info(text)
        if patient_info:
            fields.append(ParsedField(
                field_name="患者信息",
                field_name_en="patient_info",
                value=patient_info,
                confidence=0.85,
            ))

        # Extract sections
        for field_key, pattern_info in self.SECTION_PATTERNS.items():
            for keyword in pattern_info["zh"]:
                # Try to find section content
                section_content = self._extract_section(text, keyword)
                if section_content:
                    fields.append(ParsedField(
                        field_name=keyword,
                        field_name_en=pattern_info["en"],
                        value=section_content.strip(),
                        confidence=0.8,
                    ))
                    break

        return fields

    def _extract_patient_info(self, text: str) -> Optional[str]:
        """Extract patient basic information."""
        patterns = [
            r'患者[：:\s]*([^\n,，]{2,20})',
            r'姓名[：:\s]*([^\n,，]{2,10})',
            r'(\w+)\s*[，,]\s*(?:男|女)\s*[，,]\s*\d+\s*岁',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
        return None

    def _extract_section(self, text: str, keyword: str) -> Optional[str]:
        """Extract content of a specific section."""
        # Find the section header
        header_pattern = re.compile(re.escape(keyword) + r'[：:\s]*\n?')
        match = header_pattern.search(text)
        if not match:
            return None

        start = match.end()
        # Find the next section header or end of text
        next_section_pattern = re.compile(
            r'(?:' + '|'.join(
                re.escape(kw)
                for info in self.SECTION_PATTERNS.values()
                for kw in info["zh"]
                if kw != keyword
            ) + r')[：:\s]'
        )
        next_match = next_section_pattern.search(text, start)

        if next_match:
            content = text[start:next_match.start()].strip()
        else:
            content = text[start:].strip()

        # Limit content length to avoid capturing too much
        if len(content) > 500:
            content = content[:500] + "..."

        return content if content else None

    def parse_batch(self, texts: List[str]) -> List[ParsedRecord]:
        """Parse multiple records in batch."""
        return [self.parse(text) for text in texts]
