import json
import re
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    question: str
    reference: str
    prediction: str
    score: Optional[float]
    judge_response: str


def exact_match(pred: str, ref: str) -> float:
    return float(pred.strip() == ref.strip())


def _tokenize(text: str) -> list:
    import unicodedata
    tokens = []
    for ch in text:
        if unicodedata.category(ch)[0] in ("L", "N"):
            tokens.append(ch)
    return tokens


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = set(_tokenize(pred))
    ref_tokens  = set(_tokenize(ref))
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
    recall    = len(pred_tokens & ref_tokens) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    p_tok = _tokenize(pred)
    r_tok = _tokenize(ref)
    if not p_tok or not r_tok:
        return 0.0
    l = lcs(p_tok, r_tok)
    precision = l / len(p_tok)
    recall    = l / len(r_tok)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_score_from_judge(response: str) -> Optional[float]:
    patterns = [
        r"(?:スコア|score)[：:]\s*(\d+(?:\.\d+)?)",
        r"\[\[(\d+(?:\.\d+)?)\]\]",
        r"(\d+(?:\.\d+)?)\s*/\s*(?:5|10)",
        r"\b([1-5])\b",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def build_judge_prompt(question: str, reference: str, prediction: str) -> str:
    return f"""以下の質問に対する回答を評価してください。

【質問】
{question}

【参照回答】
{reference}

【評価対象の回答】
{prediction}

評価基準：
1. 正確性（回答が事実として正しいか）
2. 完全性（必要な情報が含まれているか）
3. 流暢性（自然な日本語か）

1〜5点で採点し、[[スコア]] の形式で出力してください。
例: [[3]]"""


class AutoEvaluator:
    def __init__(self, metric="rouge_l"):
        self.metric_fn = {
            "exact_match": exact_match,
            "token_f1": token_f1,
            "rouge_l": rouge_l,
        }[metric]
        self.metric_name = metric

    def evaluate(self, items: list[dict]) -> dict:
        results = []
        scores  = []
        for item in items:
            q   = item.get("question", "")
            ref = item.get("reference", "")
            pred = item.get("prediction", "")
            score = self.metric_fn(pred, ref)
            scores.append(score)
            results.append(EvalResult(q, ref, pred, score, ""))
        avg = sum(scores) / len(scores) if scores else 0.0
        return {"metric": self.metric_name, "average": avg, "results": results}


def print_report(eval_output: dict):
    print(f"\n{'='*50}")
    print(f"Metric : {eval_output['metric']}")
    print(f"Average: {eval_output['average']:.4f}")
    print(f"{'='*50}")
    for i, r in enumerate(eval_output["results"][:5]):
        print(f"\n[{i+1}] Q: {r.question[:60]}")
        print(f"     Ref : {r.reference[:60]}")
        print(f"     Pred: {r.prediction[:60]}")
        print(f"     Score: {r.score:.4f}")


if __name__ == "__main__":
    samples = [
        {"question": "富士山の高さは？", "reference": "3776メートル", "prediction": "富士山は3776メートルです"},
        {"question": "日本の首都は？",   "reference": "東京",          "prediction": "東京です"},
        {"question": "1+1は？",           "reference": "2",            "prediction": "3"},
        {"question": "Pythonとは？",      "reference": "プログラミング言語", "prediction": "Pythonはプログラミング言語の一種です"},
        {"question": "空の色は？",         "reference": "青",           "prediction": "青色"},
    ]

    for metric in ["exact_match", "token_f1", "rouge_l"]:
        ev = AutoEvaluator(metric)
        out = ev.evaluate(samples)
        print_report(out)

    print("\n=== LLM-as-a-Judge プロンプト例 ===")
    print(build_judge_prompt(
        "富士山の高さは？",
        "3776メートル",
        "富士山は日本一高い山で、3776メートルあります"
    ))
