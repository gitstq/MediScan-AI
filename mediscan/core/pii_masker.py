# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
PII (Protected Health Information) Masker.
Detects and masks sensitive personal information in clinical text.
Supports Chinese ID cards, phone numbers, names, addresses, and more.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class MaskStrategy(Enum):
    """PII masking strategies."""
    REPLACE = "replace"        # Replace with placeholder
    HASH = "hash"              # Replace with hash
    REDACT = "redact"          # Full redaction (***)
    PARTIAL = "partial"        # Partial masking (keep first/last chars)
    TAG = "tag"                # Wrap with XML-like tags


@dataclass
class PIIFound:
    """A single PII detection."""
    pii_type: str
    original_text: str
    masked_text: str
    start: int
    end: int
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "pii_type": self.pii_type,
            "original": self.original_text,
            "masked": self.masked_text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class MaskResult:
    """PII masking result."""
    masked_text: str
    pii_found: List[PIIFound] = field(default_factory=list)
    total_pii: int = 0
    pii_types: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "masked_text": self.masked_text,
            "pii_found": [p.to_dict() for p in self.pii_found],
            "total_pii": self.total_pii,
            "pii_types": self.pii_types,
        }


class PIIMasker:
    """
    PII Detection and Masking Engine.
    Supports 12+ PII types across Chinese and English text.
    Zero external dependencies - pure regex-based detection.
    """

    # Chinese surname list (top 100 common surnames)
    SURNAMES_ZH = (
        "王李张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧"
        "程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜"
        "范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱"
        "汤尹黎易常武乔贺赖龚文"
    )

    def __init__(self, strategy: MaskStrategy = MaskStrategy.REDACT):
        """
        Initialize PII masker.

        Args:
            strategy: Default masking strategy.
        """
        self.strategy = strategy
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Tuple[re.Pattern, str]]:
        """Compile all PII detection patterns."""
        patterns = {}

        # Chinese ID card (18 digits, last can be X)
        patterns["id_card_zh"] = (
            re.compile(r'(?<!\d)([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)'),
            "身份证号"
        )

        # Chinese phone numbers
        patterns["phone_zh"] = (
            re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)'),
            "手机号"
        )

        # Chinese landline
        patterns["landline_zh"] = (
            re.compile(r'(?<!\d)(0\d{2,3}[-–]?\d{7,8})(?!\d)'),
            "座机号"
        )

        # Email
        patterns["email"] = (
            re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),
            "邮箱"
        )

        # Chinese name (surname + 1-2 characters)
        patterns["name_zh"] = (
            re.compile(r'([' + self.SURNAMES_ZH + r'][\u4e00-\u9fff]{1,2})'),
            "姓名"
        )

        # IP address
        patterns["ip_address"] = (
            re.compile(r'(?<!\d)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?!\d)'),
            "IP地址"
        )

        # Chinese address keywords + patterns
        patterns["address_zh"] = (
            re.compile(r'([\u4e00-\u9fff]{2,}(?:省|市|区|县|镇|乡|村|街道|路|号|幢|栋|单元|室|楼|层|座|院|弄|巷|公寓|小区|花园|广场|大厦|中心|医院|学校|公司|集团))'),
            "地址"
        )

        # Bank card number (16-19 digits)
        patterns["bank_card"] = (
            re.compile(r'(?<!\d)(\d{16,19})(?!\d)'),
            "银行卡号"
        )

        # Passport number
        patterns["passport"] = (
            re.compile(r'(?<![A-Z])([A-Z][A-Z0-9]\d{7,8})(?![A-Z0-9])'),
            "护照号"
        )

        # Medical record number
        patterns["medical_id"] = (
            re.compile(r'(?:病历号|住院号|门诊号|患者ID)[：:\s]*(\d{6,12})'),
            "病历号"
        )

        # Date of birth
        patterns["birth_date"] = (
            re.compile(r'(?:出生日期|生日|DOB)[：:\s]*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)'),
            "出生日期"
        )

        # Insurance number
        patterns["insurance"] = (
            re.compile(r'(?:医保号|社保号|医保卡号)[：:\s]*(\d{8,20})'),
            "医保号"
        )

        return patterns

    def _mask_value(self, value: str, pii_type: str, strategy: Optional[MaskStrategy] = None) -> str:
        """Apply masking strategy to a detected PII value."""
        strat = strategy or self.strategy

        if strat == MaskStrategy.REDACT:
            return "[已脱敏]"
        elif strat == MaskStrategy.HASH:
            return hashlib.sha256(value.encode()).hexdigest()[:12]
        elif strat == MaskStrategy.PARTIAL:
            if len(value) <= 2:
                return value[0] + "*"
            elif len(value) <= 4:
                return value[0] + "*" * (len(value) - 2) + value[-1]
            else:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
        elif strat == MaskStrategy.TAG:
            return f"<PII type=\"{pii_type}\">{value}</PII>"
        else:  # REPLACE
            return f"[{pii_type}]"

    def mask(
        self,
        text: str,
        strategy: Optional[MaskStrategy] = None,
        exclude_types: Optional[List[str]] = None,
    ) -> MaskResult:
        """
        Detect and mask PII in text.

        Args:
            text: Input clinical text.
            strategy: Override default masking strategy.
            exclude_types: PII types to exclude from masking.

        Returns:
            MaskResult with masked text and PII details.
        """
        if not text or not text.strip():
            return MaskResult(masked_text=text or "")

        exclude = set(exclude_types or [])
        pii_list = []
        replacements = []  # (start, end, replacement)

        for pattern_key, (pattern, pii_label) in self._patterns.items():
            if pii_label in exclude:
                continue

            for match in pattern.finditer(text):
                matched = match.group(0)
                # For patterns with groups, use the full match for replacement
                # but the group for the actual PII value
                start = match.start()
                end = match.end()

                # Check overlap with existing detections
                overlap = False
                for existing_start, existing_end, _ in replacements:
                    if not (end <= existing_start or start >= existing_end):
                        overlap = True
                        break

                if not overlap:
                    masked = self._mask_value(matched, pii_label, strategy)
                    replacements.append((start, end, masked))
                    pii_list.append(PIIFound(
                        pii_type=pii_label,
                        original_text=matched,
                        masked_text=masked,
                        start=start,
                        end=end,
                        confidence=0.9,
                    ))

        # Apply replacements in reverse order to maintain positions
        masked_text = text
        for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            masked_text = masked_text[:start] + replacement + masked_text[end:]

        # Build type summary
        type_counts = {}
        for p in pii_list:
            type_counts[p.pii_type] = type_counts.get(p.pii_type, 0) + 1

        return MaskResult(
            masked_text=masked_text,
            pii_found=pii_list,
            total_pii=len(pii_list),
            pii_types=type_counts,
        )

    def detect(self, text: str) -> List[PIIFound]:
        """Detect PII without masking (dry run)."""
        result = self.mask(text, strategy=MaskStrategy.TAG)
        return result.pii_found
