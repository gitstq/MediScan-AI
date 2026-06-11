# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Basic Drug Interaction Checker.
Checks for common drug-drug interactions using a built-in knowledge base.
This is a simplified checker for educational/reference purposes only.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum


class InteractionSeverity(Enum):
    """Drug interaction severity levels."""
    CRITICAL = "critical"    # 严重 - contraindicated
    MAJOR = "major"          # 重要 - monitor closely
    MODERATE = "moderate"    # 中等 - consider alternatives
    MINOR = "minor"          # 轻微 - informational


@dataclass
class DrugInteraction:
    """A single drug-drug interaction."""
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    description: str
    mechanism: str = ""
    recommendation: str = ""

    def to_dict(self) -> Dict:
        return {
            "drug_a": self.drug_a,
            "drug_b": self.drug_b,
            "severity": self.severity.value,
            "description": self.description,
            "mechanism": self.mechanism,
            "recommendation": self.recommendation,
        }


@dataclass
class CheckResult:
    """Drug interaction check result."""
    drugs_checked: List[str] = field(default_factory=list)
    interactions: List[DrugInteraction] = field(default_factory=list)
    safe_pairs: int = 0
    total_pairs: int = 0

    def to_dict(self) -> Dict:
        severity_summary = {}
        for i in self.interactions:
            sev = i.severity.value
            severity_summary[sev] = severity_summary.get(sev, 0) + 1

        return {
            "drugs_checked": self.drugs_checked,
            "total_pairs_checked": self.total_pairs,
            "interactions_found": len(self.interactions),
            "safe_pairs": self.safe_pairs,
            "severity_summary": severity_summary,
            "interactions": [i.to_dict() for i in self.interactions],
            "has_critical": any(i.severity == InteractionSeverity.CRITICAL for i in self.interactions),
            "has_major": any(i.severity == InteractionSeverity.MAJOR for i in self.interactions),
        }


class DrugChecker:
    """
    Drug-Drug Interaction Checker.
    Uses a curated knowledge base of common interactions.
    For educational/reference purposes only - NOT for clinical decision making.
    """

    # Drug name aliases for matching
    DRUG_ALIASES: Dict[str, Set[str]] = {
        "阿司匹林": {"aspirin", "阿斯匹林", "拜阿司匹灵", "巴米尔"},
        "布洛芬": {"ibuprofen", "芬必得", "美林"},
        "华法林": {"warfarin", "华法令"},
        "氯吡格雷": {"clopidogrel", "波立维", "泰嘉"},
        "二甲双胍": {"metformin", "格华止", "美迪康"},
        "硝苯地平": {"nifedipine", "拜新同", "心痛定"},
        "氨氯地平": {"amlodipine", "络活喜"},
        "美托洛尔": {"metoprolol", "倍他乐克", "美托洛尔"},
        "阿托伐他汀": {"atorvastatin", "立普妥", "阿托伐他汀钙"},
        "奥美拉唑": {"omeprazole", "洛赛克", "奥美拉唑"},
        "地塞米松": {"dexamethasone", "地塞米松"},
        "胰岛素": {"insulin", "诺和灵", "优泌林"},
        "头孢": {"cephalosporin", "头孢菌素", "头孢曲松", "头孢呋辛"},
        "阿莫西林": {"amoxicillin", "阿莫仙", "弗莱莫星"},
        "甲硝唑": {"metronidazole", "甲硝唑", "灭滴灵"},
        "左氧氟沙星": {"levofloxacin", "左氧", "可乐必妥"},
        "辛伐他汀": {"simvastatin", "舒降之"},
        "卡托普利": {"captopril", "开搏通"},
        "缬沙坦": {"valsartan", "代文"},
        "泼尼松": {"prednisone", "强的松"},
        "氯雷他定": {"loratadine", "开瑞坦"},
        "西替利嗪": {"cetirizine", "仙特明"},
    }

    # Interaction knowledge base
    INTERACTIONS: List[Tuple[str, str, InteractionSeverity, str, str, str]] = [
        # (drug_a, drug_b, severity, description, mechanism, recommendation)
        (
            "阿司匹林", "华法林",
            InteractionSeverity.CRITICAL,
            "阿司匹林与华法林联用显著增加出血风险",
            "阿司匹林抑制血小板聚集，华法林抑制凝血因子，两者联用抗凝作用叠加",
            "避免联用。如必须联用，密切监测INR和出血征象",
        ),
        (
            "布洛芬", "华法林",
            InteractionSeverity.MAJOR,
            "布洛芬可能增强华法林抗凝效果，增加出血风险",
            "布洛芬可置换华法林与血浆蛋白的结合，并抑制华法林代谢",
            "如需联用，密切监测INR，考虑替代方案",
        ),
        (
            "阿司匹林", "布洛芬",
            InteractionSeverity.MODERATE,
            "布洛芬可能减弱阿司匹林的心脏保护作用",
            "布洛芬竞争性阻断阿司匹林对COX-1的不可逆抑制",
            "如需联用，建议间隔至少2小时服用",
        ),
        (
            "氯吡格雷", "奥美拉唑",
            InteractionSeverity.MAJOR,
            "奥美拉唑可能降低氯吡格雷抗血小板效果",
            "奥美拉唑抑制CYP2C19，减少氯吡格雷活性代谢物生成",
            "建议使用泮托拉唑或H2受体拮抗剂替代",
        ),
        (
            "美托洛尔", "地塞米松",
            InteractionSeverity.MODERATE,
            "地塞米松可能降低美托洛尔的降压效果",
            "糖皮质激素引起水钠潴留，拮抗β受体阻滞剂降压效果",
            "联用时需监测血压，可能需要调整剂量",
        ),
        (
            "辛伐他汀", "氨氯地平",
            InteractionSeverity.MAJOR,
            "氨氯地平可升高辛伐他汀血药浓度，增加肌病风险",
            "氨氯地平抑制CYP3A4，减慢辛伐他汀代谢",
            "辛伐他汀剂量不应超过20mg/日，或换用阿托伐他汀",
        ),
        (
            "甲硝唑", "华法林",
            InteractionSeverity.CRITICAL,
            "甲硝唑显著增强华法林抗凝效果，增加出血风险",
            "甲硝唑抑制华法林代谢酶，延长其半衰期",
            "避免联用。如必须联用，密切监测INR",
        ),
        (
            "左氧氟沙星", "布洛芬",
            InteractionSeverity.MODERATE,
            "联用可能增加癫痫发作风险",
            "氟喹诺酮类与NSAIDs联用可降低癫痫发作阈值",
            "有癫痫病史者避免联用",
        ),
        (
            "卡托普利", "螺内酯",
            InteractionSeverity.MAJOR,
            "ACEI与保钾利尿剂联用增加高钾血症风险",
            "两者均可减少钾排泄，联用可能导致严重高钾血症",
            "联用时密切监测血钾，避免补钾",
        ),
        (
            "二甲双胍", "造影剂",
            InteractionSeverity.MAJOR,
            "使用碘造影剂时二甲双胍可能增加乳酸酸中毒风险",
            "造影剂引起肾功能一过性减退，影响二甲双胍排泄",
            "造影前48h停用二甲双胍，造影后48h确认肾功能正常后恢复",
        ),
        (
            "阿托伐他汀", "克拉霉素",
            InteractionSeverity.MAJOR,
            "克拉霉素显著升高阿托伐他汀血药浓度，增加肌病风险",
            "克拉霉素强效抑制CYP3A4和P-gp",
            "联用期间阿托伐他汀剂量不超过20mg/日",
        ),
        (
            "泼尼松", "布洛芬",
            InteractionSeverity.MODERATE,
            "糖皮质激素与NSAIDs联用增加胃肠道溃疡和出血风险",
            "两者均可损伤胃黏膜，联用有协同损伤效应",
            "联用时加用胃黏膜保护剂，监测消化道症状",
        ),
    ]

    def __init__(self):
        """Initialize the drug checker."""
        self._build_lookup()

    def _build_lookup(self):
        """Build lookup structures for fast matching."""
        self._alias_to_canonical = {}
        for canonical, aliases in self.DRUG_ALIASES.items():
            self._alias_to_canonical[canonical] = canonical
            for alias in aliases:
                self._alias_to_canonical[alias.lower()] = canonical

        self._interaction_set = set()
        for drug_a, drug_b, *rest in self.INTERACTIONS:
            self._interaction_set.add((drug_a, drug_b))
            self._interaction_set.add((drug_b, drug_a))

    def _normalize_drug(self, name: str) -> Optional[str]:
        """Normalize a drug name to its canonical form."""
        name = name.strip()
        if name in self._alias_to_canonical:
            return self._alias_to_canonical[name]
        if name.lower() in self._alias_to_canonical:
            return self._alias_to_canonical[name.lower()]
        return None

    def check(self, drugs: List[str]) -> CheckResult:
        """
        Check for drug-drug interactions among a list of drugs.

        Args:
            drugs: List of drug names (Chinese or English).

        Returns:
            CheckResult with interaction details.
        """
        # Normalize drug names
        normalized = []
        for drug in drugs:
            canon = self._normalize_drug(drug)
            if canon:
                normalized.append(canon)

        # Remove duplicates
        unique_drugs = list(dict.fromkeys(normalized))

        interactions = []
        safe_pairs = 0
        total_pairs = 0

        for i in range(len(unique_drugs)):
            for j in range(i + 1, len(unique_drugs)):
                total_pairs += 1
                pair = (unique_drugs[i], unique_drugs[j])
                if pair in self._interaction_set:
                    # Find the interaction details
                    for drug_a, drug_b, severity, desc, mech, rec in self.INTERACTIONS:
                        if (drug_a == pair[0] and drug_b == pair[1]) or \
                           (drug_a == pair[1] and drug_b == pair[0]):
                            interactions.append(DrugInteraction(
                                drug_a=pair[0],
                                drug_b=pair[1],
                                severity=severity,
                                description=desc,
                                mechanism=mech,
                                recommendation=rec,
                            ))
                            break
                else:
                    safe_pairs += 1

        return CheckResult(
            drugs_checked=unique_drugs,
            interactions=interactions,
            safe_pairs=safe_pairs,
            total_pairs=total_pairs,
        )
