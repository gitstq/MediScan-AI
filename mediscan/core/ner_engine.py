# MediScan-AI - Lightweight Local Medical Text Intelligence Analysis Engine
# Copyright (c) 2026 gitstq. MIT License.

"""
Medical Named Entity Recognition Engine.
Uses rule-based + dictionary matching for zero-dependency local inference.
Supports Chinese and English clinical text.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class EntityType(Enum):
    """Medical entity types."""
    SYMPTOM = "symptom"          # 症状
    DISEASE = "disease"          # 疾病
    DRUG = "drug"                # 药品
    EXAMINATION = "examination"  # 检查项目
    TREATMENT = "treatment"      # 治疗方案
    BODY_PART = "body_part"      # 身体部位
    TEST_RESULT = "test_result"  # 检验结果
    TIME = "time"                # 时间


@dataclass
class Entity:
    """A single medical entity."""
    text: str
    entity_type: EntityType
    start: int = 0
    end: int = 0
    confidence: float = 1.0
    normalized: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized": self.normalized,
        }


@dataclass
class NERResult:
    """NER analysis result."""
    entities: List[Entity] = field(default_factory=list)
    text_length: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "total_entities": len(self.entities),
            "text_length": self.text_length,
            "processing_time_ms": self.processing_time_ms,
            "summary": self._summarize(),
        }

    def _summarize(self) -> Dict[str, int]:
        summary = {}
        for etype in EntityType:
            count = sum(1 for e in self.entities if e.entity_type == etype)
            if count > 0:
                summary[etype.value] = count
        return summary


class MedicalNER:
    """
    Rule-based Medical NER Engine.
    Zero external model dependency - uses curated dictionaries and pattern matching.
    Supports both Chinese and English clinical text.
    """

    # --- Chinese Medical Dictionaries ---

    SYMPTOMS_ZH = [
        "头痛", "头晕", "恶心", "呕吐", "发热", "咳嗽", "咳痰", "胸闷", "气短",
        "心悸", "腹痛", "腹泻", "便秘", "尿频", "尿急", "尿痛", "腰痛", "关节痛",
        "肌肉酸痛", "乏力", "失眠", "嗜睡", "食欲不振", "消瘦", "水肿", "盗汗",
        "畏寒", "鼻塞", "流涕", "咽痛", "声音嘶哑", "耳鸣", "听力下降", "视力模糊",
        "皮疹", "瘙痒", "出血", "贫血", "黄疸", "呼吸困难", "胸痛", "背痛",
        "颈部疼痛", "肩痛", "膝盖痛", "麻木", "刺痛", "灼热感", "冷汗",
        "口干", "口苦", "多饮", "多尿", "体重下降", "体重增加", "情绪低落",
        "焦虑", "烦躁", "抽搐", "昏迷", "意识模糊", "健忘",
    ]

    DISEASES_ZH = [
        "高血压", "糖尿病", "冠心病", "脑梗塞", "脑出血", "肺炎", "支气管炎",
        "哮喘", "胃炎", "胃溃疡", "十二指肠溃疡", "肝炎", "肝硬化", "脂肪肝",
        "胆囊炎", "胆结石", "肾结石", "肾炎", "肾功能不全", "尿路感染",
        "甲状腺功能亢进", "甲状腺功能减退", "贫血", "白血病", "淋巴瘤",
        "肺癌", "胃癌", "肝癌", "结肠癌", "乳腺癌", "宫颈癌", "前列腺癌",
        "骨质疏松", "关节炎", "类风湿性关节炎", "痛风", "系统性红斑狼疮",
        "过敏性鼻炎", "鼻窦炎", "中耳炎", "结膜炎", "青光眼", "白内障",
        "抑郁症", "焦虑症", "精神分裂症", "癫痫", "帕金森病", "阿尔茨海默病",
        "脑卒中", "心肌梗死", "心力衰竭", "心律失常", "心房颤动",
        "感冒", "流感", "结核病", "梅毒", "艾滋病", "乙肝", "丙肝",
        "带状疱疹", "湿疹", "荨麻疹", "银屑病",
    ]

    DRUGS_ZH = [
        "阿司匹林", "布洛芬", "对乙酰氨基酚", "头孢", "阿莫西林", "青霉素",
        "红霉素", "阿奇霉素", "左氧氟沙星", "环丙沙星", "甲硝唑", "奥硝唑",
        "二甲双胍", "格列美脲", "胰岛素", "甘精胰岛素", "阿卡波糖",
        "硝苯地平", "氨氯地平", "缬沙坦", "厄贝沙坦", "依那普利", "卡托普利",
        "美托洛尔", "比索洛尔", "阿替洛尔",
        "辛伐他汀", "阿托伐他汀", "瑞舒伐他汀", "普伐他汀",
        "氯吡格雷", "华法林", "利伐沙班", "达比加群",
        "奥美拉唑", "兰索拉唑", "泮托拉唑", "雷贝拉唑",
        "蒙脱石散", "益生菌", "乳果糖", "聚乙二醇",
        "地塞米松", "泼尼松", "甲泼尼龙",
        "氯雷他定", "西替利嗪", "扑尔敏",
        "氨溴索", "溴己新", "沙丁胺醇", "布地奈德",
        "碳酸钙", "维生素D", "钙片", "骨化三醇",
        "叶酸", "铁剂", "维生素B12",
    ]

    EXAMINATIONS_ZH = [
        "血常规", "尿常规", "便常规", "肝功能", "肾功能", "血脂", "血糖",
        "糖化血红蛋白", "凝血功能", "电解质", "甲状腺功能", "肿瘤标志物",
        "心电图", "胸片", "CT", "MRI", "B超", "彩超", "超声", "X光",
        "胃镜", "肠镜", "支气管镜", "腹腔镜", "宫腔镜",
        "血气分析", "血培养", "药敏试验", "基因检测", "病理检查",
        "眼底检查", "听力测试", "肺功能检查", "骨密度检查",
        "冠脉造影", "脑血管造影", "PET-CT", "动态心电图",
    ]

    TREATMENTS_ZH = [
        "手术", "切除术", "穿刺术", "活检", "化疗", "放疗", "靶向治疗",
        "免疫治疗", "介入治疗", "透析", "输血", "输液", "吸氧",
        "雾化吸入", "物理治疗", "康复训练", "针灸", "推拿", "理疗",
        "清创缝合", "石膏固定", "牵引", "心脏支架", "起搏器植入",
        "抗感染治疗", "对症治疗", "支持治疗", "保守治疗",
    ]

    BODY_PARTS_ZH = [
        "头部", "面部", "颈部", "胸部", "腹部", "背部", "腰部", "盆腔",
        "上肢", "下肢", "左手", "右手", "左腿", "右腿", "左眼", "右眼",
        "左耳", "右耳", "心脏", "肝脏", "脾脏", "胆囊", "胰腺", "肾脏",
        "膀胱", "子宫", "卵巢", "前列腺", "甲状腺", "肺", "胃", "肠",
        "大脑", "小脑", "脑干", "脊柱", "骨骼", "关节", "皮肤",
    ]

    # --- English Medical Dictionaries ---

    SYMPTOMS_EN = [
        "headache", "dizziness", "nausea", "vomiting", "fever", "cough", "chest pain",
        "shortness of breath", "palpitation", "abdominal pain", "diarrhea", "constipation",
        "fatigue", "insomnia", "loss of appetite", "weight loss", "swelling", "rash",
        "itching", "bleeding", "numbness", "tingling", "joint pain", "muscle pain",
        "back pain", "sore throat", "runny nose", "anxiety", "depression",
    ]

    DISEASES_EN = [
        "hypertension", "diabetes", "coronary heart disease", "pneumonia", "asthma",
        "gastritis", "hepatitis", "cirrhosis", "kidney disease", "anemia",
        "leukemia", "lung cancer", "breast cancer", "osteoporosis", "arthritis",
        "rheumatoid arthritis", "gout", "depression", "anxiety disorder",
        "epilepsy", "Parkinson's disease", "Alzheimer's disease", "stroke",
        "myocardial infarction", "heart failure", "arrhythmia", "atrial fibrillation",
        "influenza", "tuberculosis", "HIV", "herpes", "eczema", "psoriasis",
    ]

    DRUGS_EN = [
        "aspirin", "ibuprofen", "acetaminophen", "amoxicillin", "penicillin",
        "azithromycin", "metformin", "insulin", "amlodipine", "valsartan",
        "metoprolol", "atorvastatin", "clopidogrel", "warfarin", "omeprazole",
        "prednisone", "loratadine", " cetirizine", "salbutamol",
    ]

    EXAMINATIONS_EN = [
        "blood test", "urinalysis", "liver function", "kidney function",
        "blood lipid", "blood glucose", "ECG", "X-ray", "CT scan", "MRI",
        "ultrasound", "endoscopy", "colonoscopy", "biopsy", "pathology",
    ]

    def __init__(self, language: str = "auto"):
        """
        Initialize NER engine.

        Args:
            language: "zh" for Chinese, "en" for English, "auto" for auto-detect.
        """
        self.language = language
        self._jieba_initialized = False

        # Compile all dictionaries into lookup structures
        self._dict_zh = self._build_dictionary(
            self.SYMPTOMS_ZH, self.DISEASES_ZH, self.DRUGS_ZH,
            self.EXAMINATIONS_ZH, self.TREATMENTS_ZH, self.BODY_PARTS_ZH
        )
        self._dict_en = self._build_dictionary(
            self.SYMPTOMS_EN, self.DISEASES_EN, self.DRUGS_EN,
            self.EXAMINATIONS_EN, [], []
        )

        # Compile regex patterns for test results
        self._result_patterns = self._compile_result_patterns()

    def _build_dictionary(self, symptoms, diseases, drugs, examinations, treatments, body_parts) -> Dict:
        """Build a dictionary mapping entity type to sorted term list (longest first)."""
        return {
            EntityType.SYMPTOM: sorted(symptoms, key=len, reverse=True),
            EntityType.DISEASE: sorted(diseases, key=len, reverse=True),
            EntityType.DRUG: sorted(drugs, key=len, reverse=True),
            EntityType.EXAMINATION: sorted(examinations, key=len, reverse=True),
            EntityType.TREATMENT: sorted(treatments, key=len, reverse=True),
            EntityType.BODY_PART: sorted(body_parts, key=len, reverse=True),
        }

    def _compile_result_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for detecting test results."""
        patterns = [
            # Chinese patterns: 数值 + 单位
            re.compile(r'(\d+\.?\d*)\s*(mmHg|mg/dL|g/L|U/L|pg/mL|ng/mL|μmol/L|mmol/L|×10⁹/L|×10¹²/L|%|次/分|次/分|bpm)'),
            # English patterns
            re.compile(r'(\d+\.?\d*)\s*(mg/dL|g/L|U/L|pg/mL|ng/mL|umol/L|mmol/L|mmHg|%|bpm)'),
            # Abnormal markers
            re.compile(r'(偏高|偏低|异常|升高|降低|增高|减少|阳性|阴性|positive|negative|high|low|elevated|abnormal)'),
        ]
        return patterns

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English."""
        zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if zh_chars / max(len(text), 1) > 0.1:
            return "zh"
        return "en"

    def _ensure_jieba(self):
        """Lazy-initialize jieba for Chinese segmentation."""
        if not self._jieba_initialized:
            try:
                import jieba
                jieba.setLogLevel(20)  # Suppress logging
                self._jieba_initialized = True
            except ImportError:
                self._jieba_initialized = False

    def analyze(self, text: str, language: Optional[str] = None) -> NERResult:
        """
        Analyze medical text and extract named entities.

        Args:
            text: Clinical text to analyze.
            language: Override language detection ("zh", "en", or None for auto).

        Returns:
            NERResult with extracted entities.
        """
        import time
        start_time = time.time()

        if not text or not text.strip():
            return NERResult(text_length=0)

        lang = language or self.language
        if lang == "auto":
            lang = self._detect_language(text)

        entities = []
        dictionary = self._dict_zh if lang == "zh" else self._dict_en

        # Dictionary-based matching (longest match first)
        for entity_type, terms in dictionary.items():
            for term in terms:
                idx = 0
                while True:
                    pos = text.find(term, idx)
                    if pos == -1:
                        break
                    # Check for overlapping entities
                    overlap = False
                    for existing in entities:
                        if not (pos >= existing.end or pos + len(term) <= existing.start):
                            overlap = True
                            break
                    if not overlap:
                        entities.append(Entity(
                            text=term,
                            entity_type=entity_type,
                            start=pos,
                            end=pos + len(term),
                            confidence=0.9 if lang == "zh" else 0.85,
                        ))
                    idx = pos + 1

        # Regex-based test result detection
        for pattern in self._result_patterns:
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                pos = match.start()
                overlap = False
                for existing in entities:
                    if not (pos >= existing.end or pos + len(matched_text) <= existing.start):
                        overlap = True
                        break
                if not overlap:
                    entities.append(Entity(
                        text=matched_text,
                        entity_type=EntityType.TEST_RESULT,
                        start=pos,
                        end=pos + len(matched_text),
                        confidence=0.75,
                    ))

        # Sort entities by position
        entities.sort(key=lambda e: e.start)

        elapsed = (time.time() - start_time) * 1000

        return NERResult(
            entities=entities,
            text_length=len(text),
            processing_time_ms=round(elapsed, 2),
        )

    def analyze_batch(self, texts: List[str], language: Optional[str] = None) -> List[NERResult]:
        """Analyze multiple texts in batch."""
        return [self.analyze(text, language) for text in texts]
