# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Unit tests for PIIMasker.
"""

import pytest
from mediscan.core.pii_masker import PIIMasker, MaskStrategy, MaskResult


@pytest.fixture
def masker():
    """Create a PIIMasker with default redact strategy."""
    return PIIMasker(strategy=MaskStrategy.REDACT)


class TestIDCardMasking:
    """Test Chinese ID card number masking."""

    def test_valid_id_card(self, masker):
        """Test masking of a valid 18-digit Chinese ID card."""
        text = "身份证号110101199001011234"
        result = masker.mask(text)
        assert result.total_pii >= 1
        assert "[已脱敏]" in result.masked_text
        pii_types = [p.pii_type for p in result.pii_found]
        assert "身份证号" in pii_types

    def test_id_card_with_x(self, masker):
        """Test masking of ID card ending with X."""
        text = "身份证号11010119900101123X"
        result = masker.mask(text)
        assert result.total_pii >= 1
        assert "身份证号" in [p.pii_type for p in result.pii_found]

    def test_id_card_lowercase_x(self, masker):
        """Test masking of ID card ending with lowercase x."""
        text = "身份证号11010119900101123x"
        result = masker.mask(text)
        assert result.total_pii >= 1

    def test_invalid_id_card_not_masked(self, masker):
        """Test that invalid ID card numbers are not masked."""
        text = "身份证号12345"
        result = masker.mask(text)
        id_cards = [p for p in result.pii_found if p.pii_type == "身份证号"]
        assert len(id_cards) == 0


class TestPhoneMasking:
    """Test phone number masking."""

    def test_mobile_phone(self, masker):
        """Test masking of Chinese mobile phone number."""
        text = "联系电话13800138000"
        result = masker.mask(text)
        assert result.total_pii >= 1
        assert "手机号" in [p.pii_type for p in result.pii_found]

    def test_phone_13x(self, masker):
        """Test masking of 13x phone numbers."""
        text = "手机号13912345678"
        result = masker.mask(text)
        phones = [p for p in result.pii_found if p.pii_type == "手机号"]
        assert len(phones) >= 1

    def test_phone_not_in_longer_number(self, masker):
        """Test that phone number is not falsely detected in longer digit strings."""
        text = "编号1234567890123456"
        result = masker.mask(text)
        phones = [p for p in result.pii_found if p.pii_type == "手机号"]
        assert len(phones) == 0

    def test_landline(self, masker):
        """Test masking of Chinese landline number."""
        text = "座机010-12345678"
        result = masker.mask(text)
        assert "座机号" in [p.pii_type for p in result.pii_found]


class TestEmailMasking:
    """Test email address masking."""

    def test_standard_email(self, masker):
        """Test masking of standard email address."""
        text = "邮箱test@example.com"
        result = masker.mask(text)
        assert result.total_pii >= 1
        assert "邮箱" in [p.pii_type for p in result.pii_found]

    def test_email_with_dots(self, masker):
        """Test masking of email with dots in name."""
        text = "发送至first.last@domain.co.cn"
        result = masker.mask(text)
        assert "邮箱" in [p.pii_type for p in result.pii_found]

    def test_no_false_positive_email(self, masker):
        """Test that non-email text is not falsely detected."""
        text = "患者无邮箱"
        result = masker.mask(text)
        emails = [p for p in result.pii_found if p.pii_type == "邮箱"]
        assert len(emails) == 0


class TestNameMasking:
    """Test Chinese name masking."""

    def test_common_name(self, masker):
        """Test masking of common Chinese name."""
        text = "患者张三"
        result = masker.mask(text)
        assert "姓名" in [p.pii_type for p in result.pii_found]

    def test_three_char_name(self, masker):
        """Test masking of three-character Chinese name."""
        text = "患者张三丰"
        result = masker.mask(text)
        assert "姓名" in [p.pii_type for p in result.pii_found]

    def test_name_with_title(self, masker):
        """Test name detection with title prefix."""
        text = "王医生诊断"
        result = masker.mask(text)
        names = [p for p in result.pii_found if p.pii_type == "姓名"]
        # "王医" might match as surname + char, but "王医生" is 3 chars
        assert len(names) >= 1


class TestAddressMasking:
    """Test Chinese address masking."""

    def test_full_address(self, masker):
        """Test masking of full Chinese address."""
        text = "住址北京市朝阳区建国路88号"
        result = masker.mask(text)
        assert "地址" in [p.pii_type for p in result.pii_found]

    def test_address_with_community(self, masker):
        """Test masking of address with community name."""
        text = "家住北京市朝阳区幸福花园小区"
        result = masker.mask(text)
        assert "地址" in [p.pii_type for p in result.pii_found]

    def test_address_with_hospital(self, masker):
        """Test masking of address with hospital."""
        text = "到北京协和医院就诊"
        result = masker.mask(text)
        assert "地址" in [p.pii_type for p in result.pii_found]


class TestMaskingStrategies:
    """Test different masking strategies."""

    def test_redact_strategy(self, masker):
        """Test redact strategy replaces with [已脱敏]."""
        text = "手机号13800138000"
        result = masker.mask(text, strategy=MaskStrategy.REDACT)
        assert "[已脱敏]" in result.masked_text

    def test_hash_strategy(self):
        """Test hash strategy replaces with SHA256 hash prefix."""
        masker = PIIMasker(strategy=MaskStrategy.HASH)
        text = "手机号13800138000"
        result = masker.mask(text, strategy=MaskStrategy.HASH)
        # Hash should be a 12-char hex string
        assert "13800138000" not in result.masked_text
        # The masked text should contain a hash-like string
        found = [p for p in result.pii_found if p.pii_type == "手机号"]
        assert len(found) == 1
        assert len(found[0].masked_text) == 12

    def test_partial_strategy_short(self):
        """Test partial strategy for short values (2 chars)."""
        masker = PIIMasker(strategy=MaskStrategy.PARTIAL)
        text = "患者张三"
        result = masker.mask(text, strategy=MaskStrategy.PARTIAL)
        # Name should be partially masked
        names = [p for p in result.pii_found if p.pii_type == "姓名"]
        assert len(names) >= 1
        assert "*" in names[0].masked_text

    def test_partial_strategy_phone(self):
        """Test partial strategy for phone numbers."""
        masker = PIIMasker(strategy=MaskStrategy.PARTIAL)
        text = "手机号13800138000"
        result = masker.mask(text, strategy=MaskStrategy.PARTIAL)
        phones = [p for p in result.pii_found if p.pii_type == "手机号"]
        assert len(phones) == 1
        # Phone: first 2 + stars + last 2
        masked = phones[0].masked_text
        assert masked.startswith("13")
        assert masked.endswith("00")
        assert "*" in masked

    def test_tag_strategy(self):
        """Test tag strategy wraps PII with XML-like tags."""
        masker = PIIMasker(strategy=MaskStrategy.TAG)
        text = "手机号13800138000"
        result = masker.mask(text, strategy=MaskStrategy.TAG)
        assert '<PII type="手机号">13800138000</PII>' in result.masked_text

    def test_replace_strategy(self):
        """Test replace strategy substitutes with type name."""
        masker = PIIMasker(strategy=MaskStrategy.REPLACE)
        text = "手机号13800138000"
        result = masker.mask(text, strategy=MaskStrategy.REPLACE)
        assert "[手机号]" in result.masked_text


class TestExcludeTypes:
    """Test excluding specific PII types from masking."""

    def test_exclude_name(self, masker):
        """Test excluding name from masking."""
        text = "患者张三，手机号13800138000"
        result = masker.mask(text, exclude_types=["姓名"])
        names = [p for p in result.pii_found if p.pii_type == "姓名"]
        assert len(names) == 0
        phones = [p for p in result.pii_found if p.pii_type == "手机号"]
        assert len(phones) >= 1

    def test_exclude_multiple_types(self, masker):
        """Test excluding multiple PII types."""
        text = "患者张三，手机号13800138000，邮箱test@example.com"
        result = masker.mask(text, exclude_types=["姓名", "邮箱"])
        pii_types = {p.pii_type for p in result.pii_found}
        assert "姓名" not in pii_types
        assert "邮箱" not in pii_types
        assert "手机号" in pii_types

    def test_exclude_all(self, masker):
        """Test excluding all PII types returns original text."""
        text = "患者张三，手机号13800138000"
        result = masker.mask(text, exclude_types=["姓名", "手机号", "座机号", "邮箱",
            "身份证号", "地址", "IP地址", "银行卡号", "护照号", "病历号", "出生日期", "医保号"])
        assert result.total_pii == 0
        assert result.masked_text == text


class TestEmptyText:
    """Test handling of empty text."""

    def test_empty_string(self, masker):
        """Test empty string returns empty result."""
        result = masker.mask("")
        assert isinstance(result, MaskResult)
        assert result.masked_text == ""
        assert result.total_pii == 0

    def test_whitespace_only(self, masker):
        """Test whitespace-only string."""
        result = masker.mask("   \n\t  ")
        assert result.total_pii == 0

    def test_none_like(self, masker):
        """Test that empty input is handled gracefully."""
        result = masker.mask("")
        assert result.pii_found == []
        assert result.pii_types == {}


class TestMaskResultStructure:
    """Test MaskResult data structure."""

    def test_result_dict(self, masker):
        """Test that to_dict produces correct structure."""
        text = "手机号13800138000"
        result = masker.mask(text)
        d = result.to_dict()
        assert "masked_text" in d
        assert "pii_found" in d
        assert "total_pii" in d
        assert "pii_types" in d
        assert d["total_pii"] == len(result.pii_found)

    def test_pii_found_dict(self, masker):
        """Test that PIIFound to_dict produces correct structure."""
        text = "手机号13800138000"
        result = masker.mask(text)
        if result.pii_found:
            p = result.pii_found[0]
            d = p.to_dict()
            assert "pii_type" in d
            assert "original" in d
            assert "masked" in d
            assert "start" in d
            assert "end" in d
            assert "confidence" in d


class TestDetectMethod:
    """Test the detect (dry run) method."""

    def test_detect_returns_pii_list(self, masker):
        """Test detect returns list without masking."""
        text = "手机号13800138000"
        pii_list = masker.detect(text)
        assert len(pii_list) >= 1
        assert pii_list[0].pii_type == "手机号"
        # detect should use TAG strategy internally
        assert "<PII" in pii_list[0].masked_text


class TestMultiplePII:
    """Test text with multiple PII types."""

    def test_complex_clinical_text(self, masker):
        """Test masking of complex clinical text with multiple PII types."""
        text = "患者王五，男，身份证号110101199001011234，手机号13800138000，邮箱wang@test.com，住址北京市海淀区中关村大街1号"
        result = masker.mask(text)
        pii_types = {p.pii_type for p in result.pii_found}
        assert result.total_pii >= 4
        assert "姓名" in pii_types
        assert "身份证号" in pii_types
        assert "手机号" in pii_types
        assert "邮箱" in pii_types
