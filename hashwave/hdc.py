from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .hypervector import HyperVector, bundle

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

_PHRASE_SYNONYMS = {
    "رمز ارز": "توکن",
    "غیر مثلی": "انفتی",
    "ان اف تی": "انفتی",
    "رای گیری": "رایگیری",
}

_WORD_SYNONYMS = {
    "سکه": "توکن",
    "ارز": "توکن",
    "رمزارز": "توکن",
    "غیرمثلی": "انفتی",
    "ان‌اف‌تی": "انفتی",
    "اثر": "دارایی",
    "آثار": "دارایی",
    "تصویر": "دارایی",
    "تصاویر": "دارایی",
    "عکس": "دارایی",
    "عکسها": "دارایی",
    "رأی": "رای",
    "آرا": "رای",
    "نظرسنجی": "رایگیری",
    "محدود": "ثابت",
    "نهایی": "ثابت",
    "سقف": "ثابت",
    "بسوزاند": "سوزاندن",
}



def normalize_fa(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Remove Arabic/Persian vowel marks instead of turning them into spaces.
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = (text.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه")
            .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ؤ", "و").replace("ئ", "ی"))
    text = text.replace("\u200c", " ").replace("ـ", "")
    text = _PUNCT_RE.sub(" ", text.lower())
    text = _SPACE_RE.sub(" ", text).strip()
    for source in sorted(_PHRASE_SYNONYMS, key=len, reverse=True):
        text = text.replace(source, _PHRASE_SYNONYMS[source])
    tokens = [_WORD_SYNONYMS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_fa(text).split() if token]


def _feature_labels(tokens: Sequence[str]) -> list[str]:
    if not tokens:
        return ["EMPTY"]
    labels: list[str] = []
    for token in tokens:
        labels.append(f"W:{token}")
        padded = f"^{token}$"
        for n in (2, 3, 4):
            labels.extend(f"C{n}:{padded[i:i+n]}" for i in range(max(0, len(padded) - n + 1)))
    labels.extend(f"B:{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    return labels


def encode_text(text_or_tokens: str | Sequence[str], dimension: int = 8192) -> HyperVector:
    tokens = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else [normalize_fa(t) for t in text_or_tokens]
    features = _feature_labels(tokens)
    vectors = [HyperVector.from_label(feature, dimension) for feature in features]
    return bundle(vectors)


@dataclass
class AssociativeClassifier:
    dimension: int = 8192
    _examples: dict[str, list[HyperVector]] = field(default_factory=dict)
    _prototypes: dict[str, HyperVector] = field(default_factory=dict)

    def add(self, label: str, text_or_tokens: str | Sequence[str]) -> None:
        if not label:
            raise ValueError("label must not be empty")
        self._examples.setdefault(label, []).append(encode_text(text_or_tokens, self.dimension))
        self._prototypes.pop(label, None)

    def fit(self) -> None:
        if not self._examples:
            raise ValueError("no training examples")
        self._prototypes = {label: bundle(vectors) for label, vectors in self._examples.items()}

    def predict(self, text_or_tokens: str | Sequence[str]) -> tuple[str, dict[str, float]]:
        if not self._prototypes:
            self.fit()
        query = encode_text(text_or_tokens, self.dimension)
        scores = {label: query.similarity(proto) for label, proto in self._prototypes.items()}
        winner = max(sorted(scores), key=lambda label: scores[label])
        return winner, scores


def build_demo_classifier() -> AssociativeClassifier:
    model = AssociativeClassifier(dimension=8192)
    examples = {
        "token": [
            "یک توکن با عرضه ثابت بساز",
            "توکن قابل سوزاندن می خواهم",
            "عرضه توکن افزایش پیدا نکند",
            "قرارداد ارز دیجیتال با سقف مشخص",
            "سکه با تعداد محدود ایجاد کن",
            "رمزارز بدون امکان ضرب مجدد",
        ],
        "nft": [
            "یک مجموعه ان اف تی بساز",
            "تصاویر را به توکن غیرمثلی تبدیل کن",
            "قرارداد مجموعه آثار دیجیتال",
            "برای هر اثر یک شناسه یکتا ایجاد کن",
            "مالکیت عکس ها را روی زنجیره ثبت کن",
            "دارایی دیجیتال غیر قابل تعویض بساز",
        ],
        "voting": [
            "یک سامانه رای گیری بساز",
            "اعضا درباره پیشنهادها رای بدهند",
            "قرارداد تصمیم گیری جمعی",
            "هر عضو یک حق رای داشته باشد",
            "نتیجه با شمارش آرا مشخص شود",
            "نظرسنجی غیر متمرکز ایجاد کن",
        ],
    }
    for label, texts in examples.items():
        for text in texts:
            model.add(label, text)
    model.fit()
    return model


def demo_classifier() -> dict:
    model = build_demo_classifier()
    query = "توکن با عرضه محدود و قابلیت سوزاندن"
    label, scores = model.predict(query)
    return {"query": query, "prediction": label, "scores": scores}
