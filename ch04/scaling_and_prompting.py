import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def scaling_law(N, A=10.0, alpha=0.076, B=1.69):
    return A / (N ** alpha) + B


def plot_scaling_law():
    params = np.logspace(7, 11, 200)
    loss = scaling_law(params)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(params, loss, linewidth=2, color="#2563eb", label=r"$L = A/N^{\alpha} + B$")

    markers = [1e7, 1e8, 1e9, 1e10, 1e11]
    for n in markers:
        ax.scatter(n, scaling_law(n), color="#dc2626", zorder=5, s=60)
        ax.annotate(
            f"{n:.0e}\nL={scaling_law(n):.2f}",
            xy=(n, scaling_law(n)),
            xytext=(n * 1.8, scaling_law(n) + 0.05),
            fontsize=7,
            color="#dc2626",
        )

    ax.set_xlabel("Parameters (N)", fontsize=12)
    ax.set_ylabel("Loss (L)", fontsize=12)
    ax.set_title("Neural Scaling Law: Loss vs. Model Parameters", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    outpath = os.path.join(SCRIPT_DIR, "scaling_law.png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"saved: {outpath}")
    return outpath


ZERO_SHOT_TEMPLATE = """\
Classify the sentiment of the following text as positive or negative.

Text: {text}
Sentiment:"""

ONE_SHOT_TEMPLATE = """\
Classify the sentiment of the following text as positive or negative.

Example:
Text: This movie was absolutely wonderful!
Sentiment: positive

Text: {text}
Sentiment:"""

FEW_SHOT_TEMPLATE = """\
Classify the sentiment of the following text as positive or negative.

Examples:
Text: This movie was absolutely wonderful!
Sentiment: positive

Text: The food was cold and the service was terrible.
Sentiment: negative

Text: It was okay, nothing special.
Sentiment: neutral

Text: {text}
Sentiment:"""

COT_TEMPLATE = """\
Solve the following problem step by step.

Problem: {problem}

Let me think through this carefully:
Step 1:"""


def demo_prompting():
    print("=" * 60)
    print("Prompting Patterns Demo")
    print("=" * 60)

    test_text = "I waited two hours and the package never arrived."
    problem = "If a train travels at 60 km/h for 2.5 hours, how far does it go?"

    print("\n--- Zero-shot ---")
    print(ZERO_SHOT_TEMPLATE.format(text=test_text))

    print("\n--- One-shot ---")
    print(ONE_SHOT_TEMPLATE.format(text=test_text))

    print("\n--- Few-shot ---")
    print(FEW_SHOT_TEMPLATE.format(text=test_text))

    print("\n--- Chain-of-Thought ---")
    print(COT_TEMPLATE.format(problem=problem))


def print_scaling_stats():
    print("=" * 60)
    print("Scaling Law Statistics")
    print("=" * 60)
    configs = [
        ("1e7  (10M)",    1e7),
        ("1e8  (100M)",   1e8),
        ("1e9  (1B)",     1e9),
        ("1e10 (10B)",    1e10),
        ("1e11 (100B)",   1e11),
    ]
    for label, n in configs:
        loss = scaling_law(n)
        pct_improvement = (scaling_law(configs[0][1]) - loss) / scaling_law(configs[0][1]) * 100
        print(f"  N={label}  ->  L={loss:.4f}  ({pct_improvement:.1f}% better than 10M)")


if __name__ == "__main__":
    plot_scaling_law()
    print_scaling_stats()
    demo_prompting()
