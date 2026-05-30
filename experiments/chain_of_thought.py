"""
Chain-of-Thought Prompting Analysis
Compares three prompting strategies on arithmetic reasoning:
  1. Direct answer
  2. Zero-shot CoT ("Let's think step by step.")
  3. Few-shot CoT (with worked examples)

"Model responses" are simulated via deterministic computation + 20% error injection.
Results saved to cot_results.json.
"""
import json
import os
import random


# ─────────────────────────── problem generation ──────────────────────────────

def generate_problems(n: int = 20, seed: int = 42) -> list[dict]:
    """
    Create 2-step (n//2) and 3-step (n//2) arithmetic problems.
    Each problem stores the exact steps so we can verify correctness.
    """
    rng = random.Random(seed)
    problems = []

    for i in range(n // 2):
        a, b, c = rng.randint(2, 30), rng.randint(2, 20), rng.randint(2, 15)
        op1 = rng.choice(["*", "+"])
        step1 = a * b if op1 == "*" else a + b
        answer = step1 + c
        question = f"What is {a} {op1} {b} + {c}?"
        steps = [
            f"First compute {a} {op1} {b} = {step1}",
            f"Then {step1} + {c} = {answer}",
        ]
        problems.append({"id": i, "type": "2-step", "question": question,
                         "steps": steps, "answer": answer})

    for i in range(n // 2):
        a, b, c, d = (rng.randint(2, 20), rng.randint(2, 15),
                      rng.randint(2, 10), rng.randint(1, 10))
        step1 = a * b
        step2 = step1 + c
        answer = step2 * d
        question = f"What is ({a} * {b} + {c}) * {d}?"
        steps = [
            f"First compute {a} * {b} = {step1}",
            f"Then {step1} + {c} = {step2}",
            f"Then {step2} * {d} = {answer}",
        ]
        problems.append({"id": n // 2 + i, "type": "3-step", "question": question,
                         "steps": steps, "answer": answer})

    rng.shuffle(problems)
    return problems


# ─────────────────────────── simulated model responses ───────────────────────

def inject_error(value: int, rng: random.Random, error_prob: float = 0.20) -> int:
    """Return a wrong answer with probability error_prob."""
    if rng.random() < error_prob:
        offset = rng.randint(1, max(1, abs(value) // 4 + 2))
        return value + rng.choice([-1, 1]) * offset
    return value


# ── Few-shot CoT examples shown in the prompt ────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is 7 * 3 + 5?",
        "reasoning": "First compute 7 * 3 = 21. Then 21 + 5 = 26.",
        "answer": 26,
    },
    {
        "question": "What is 4 + 6 * 2?",
        "reasoning": "First compute 6 * 2 = 12. Then 4 + 12 = 16.",
        "answer": 16,
    },
    {
        "question": "What is (3 * 4 + 2) * 5?",
        "reasoning": "First compute 3 * 4 = 12. Then 12 + 2 = 14. Then 14 * 5 = 70.",
        "answer": 70,
    },
]


class SimulatedModel:
    """
    Simulates LLM responses for arithmetic problems.
    - Direct strategy: higher error probability (no reasoning scaffolding).
    - Zero-shot CoT: reduced error probability (thinking prompt helps).
    - Few-shot CoT: lowest error probability (examples anchor reasoning).
    """

    STRATEGY_ERROR_RATES = {
        "direct":       0.35,   # no reasoning cue → more errors
        "zero_shot_cot": 0.18,  # "let's think step by step" helps
        "few_shot_cot":  0.08,  # worked examples help most
    }

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def respond(self, problem: dict, strategy: str) -> dict:
        error_prob = self.STRATEGY_ERROR_RATES[strategy]
        correct_answer = problem["answer"]
        predicted = inject_error(correct_answer, self.rng, error_prob)
        is_correct = predicted == correct_answer

        response = {"strategy": strategy, "predicted": predicted,
                    "correct": is_correct, "question": problem["question"],
                    "true_answer": correct_answer}

        if strategy == "direct":
            response["prompt_summary"] = problem["question"]
            response["model_output"] = str(predicted)

        elif strategy == "zero_shot_cot":
            response["prompt_summary"] = (
                f"{problem['question']} Let's think step by step."
            )
            # Simulate step-by-step reasoning (correct steps, possibly wrong conclusion)
            reasoning_steps = list(problem["steps"])
            response["model_output"] = (
                " ".join(reasoning_steps) + f" Therefore the answer is {predicted}."
            )

        elif strategy == "few_shot_cot":
            ex_lines = []
            for ex in FEW_SHOT_EXAMPLES:
                ex_lines.append(
                    f"Q: {ex['question']} A: {ex['reasoning']} Answer: {ex['answer']}"
                )
            ex_block = " | ".join(ex_lines)
            response["prompt_summary"] = f"[3 examples] {ex_block} | Q: {problem['question']}"
            reasoning_steps = list(problem["steps"])
            response["model_output"] = (
                " ".join(reasoning_steps) + f" Answer: {predicted}."
            )

        return response


# ─────────────────────────── evaluation ──────────────────────────────────────

def evaluate(responses: list[dict]) -> dict:
    by_strategy: dict[str, dict] = {}
    for r in responses:
        s = r["strategy"]
        if s not in by_strategy:
            by_strategy[s] = {"total": 0, "correct": 0,
                              "two_step_correct": 0, "three_step_correct": 0,
                              "two_step_total": 0, "three_step_total": 0}
        by_strategy[s]["total"] += 1
        by_strategy[s]["correct"] += int(r["correct"])

    return by_strategy


def accuracy(stats: dict) -> float:
    return stats["correct"] / stats["total"] if stats["total"] else 0.0


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Chain-of-Thought Prompting Analysis")
    print("=" * 60)

    problems = generate_problems(n=20)
    model = SimulatedModel(seed=7)
    strategies = ["direct", "zero_shot_cot", "few_shot_cot"]

    all_responses: list[dict] = []
    for prob in problems:
        for strategy in strategies:
            resp = model.respond(prob, strategy)
            resp["problem_id"] = prob["id"]
            resp["problem_type"] = prob["type"]
            all_responses.append(resp)

    # ── per-strategy accuracy ─────────────────────────────────────────────────
    stats: dict[str, dict] = {s: {"total": 0, "correct": 0,
                                   "2step_total": 0, "2step_correct": 0,
                                   "3step_total": 0, "3step_correct": 0}
                               for s in strategies}
    for r in all_responses:
        s = r["strategy"]
        t = r["problem_type"]
        stats[s]["total"] += 1
        stats[s]["correct"] += int(r["correct"])
        key = "2step" if t == "2-step" else "3step"
        stats[s][f"{key}_total"] += 1
        stats[s][f"{key}_correct"] += int(r["correct"])

    strategy_labels = {
        "direct":       "Direct",
        "zero_shot_cot": "Zero-shot CoT",
        "few_shot_cot":  "Few-shot CoT",
    }

    print("\n── Accuracy by Strategy ──")
    print(f"{'Strategy':<20} {'Overall':>8} {'2-step':>8} {'3-step':>8}")
    print("-" * 48)
    for s in strategies:
        st = stats[s]
        overall  = st["correct"] / st["total"] * 100
        two_step = st["2step_correct"] / st["2step_total"] * 100 if st["2step_total"] else 0
        three_step = st["3step_correct"] / st["3step_total"] * 100 if st["3step_total"] else 0
        print(f"{strategy_labels[s]:<20} {overall:>7.1f}% {two_step:>7.1f}% {three_step:>7.1f}%")

    # CoT improvement
    direct_acc    = stats["direct"]["correct"] / stats["direct"]["total"]
    few_shot_acc  = stats["few_shot_cot"]["correct"] / stats["few_shot_cot"]["total"]
    print(f"\nCoT improvement (direct→few-shot): "
          f"{(few_shot_acc - direct_acc)*100:+.1f} percentage points")

    print("\n── Sample Responses (3 examples) ──")
    seen_strategies: set = set()
    for r in all_responses:
        if r["strategy"] not in seen_strategies and not r["correct"]:
            seen_strategies.add(r["strategy"])
            label = strategy_labels[r["strategy"]]
            print(f"\n[{label}]")
            print(f"  Q: {r['question']}")
            print(f"  Model output: {r['model_output']}")
            print(f"  True answer: {r['true_answer']}  Predicted: {r['predicted']}  ✗")
        if len(seen_strategies) == 3:
            break

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "summary": {
            s: {
                "accuracy": stats[s]["correct"] / stats[s]["total"],
                "two_step_accuracy": (stats[s]["2step_correct"] / stats[s]["2step_total"]
                                      if stats[s]["2step_total"] else None),
                "three_step_accuracy": (stats[s]["3step_correct"] / stats[s]["3step_total"]
                                        if stats[s]["3step_total"] else None),
            }
            for s in strategies
        },
        "responses": all_responses,
        "n_problems": len(problems),
        "error_injection_rates": SimulatedModel.STRATEGY_ERROR_RATES,
    }

    out_path = os.path.join(os.path.dirname(__file__), "cot_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
