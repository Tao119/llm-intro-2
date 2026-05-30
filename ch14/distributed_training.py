import os
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEEPSPEED_ZERO2_CONFIG = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 100,
        },
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "steps_per_print": 50,
    "wall_clock_breakdown": False,
}

DEEPSPEED_ZERO3_CONFIG = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 100,
        },
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 2,
    "steps_per_print": 50,
    "wall_clock_breakdown": False,
}


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, nhead=4, num_layers=2, max_seq=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.encoder(h)
        return self.head(h)


class RandomTextDataset(Dataset):
    def __init__(self, size=1000, seq_len=64, vocab_size=1000):
        self.data = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


def data_parallel_training_demo():
    print("\n=== DataParallel Training Demo ===")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA GPUs available. Running single-device demo on CPU/MPS.")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model = SimpleTransformer().to(device)
    else:
        device = torch.device("cuda")
        model = SimpleTransformer().to(device)
        if num_gpus > 1:
            model = nn.DataParallel(model)
            print(f"Using DataParallel across {num_gpus} GPUs")
        else:
            print(f"Single GPU training")

    dataset = RandomTextDataset(size=500, seq_len=33)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    for epoch in range(2):
        total_loss = 0.0
        steps = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, 1000), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(f"  Epoch {epoch+1} loss: {total_loss/steps:.4f}")

    print("DataParallel training demo complete.")
    return total_params


def zero_stage_memory_analysis(total_params_million, num_gpus=8, bytes_per_param=4):
    adam_states_per_param = 3
    grad_per_param = 1
    total_mb = total_params_million * 1e6 * bytes_per_param / (1024**2)

    stages = {}

    stages["ZeRO-0"] = {
        "params_per_gpu": total_mb,
        "grads_per_gpu": total_mb * grad_per_param,
        "optimizer_per_gpu": total_mb * adam_states_per_param,
        "total_per_gpu": total_mb * (1 + grad_per_param + adam_states_per_param),
    }

    stages["ZeRO-1"] = {
        "params_per_gpu": total_mb,
        "grads_per_gpu": total_mb * grad_per_param,
        "optimizer_per_gpu": total_mb * adam_states_per_param / num_gpus,
        "total_per_gpu": total_mb * (1 + grad_per_param) + total_mb * adam_states_per_param / num_gpus,
    }

    stages["ZeRO-2"] = {
        "params_per_gpu": total_mb,
        "grads_per_gpu": total_mb * grad_per_param / num_gpus,
        "optimizer_per_gpu": total_mb * adam_states_per_param / num_gpus,
        "total_per_gpu": total_mb + (total_mb * (grad_per_param + adam_states_per_param)) / num_gpus,
    }

    stages["ZeRO-3"] = {
        "params_per_gpu": total_mb / num_gpus,
        "grads_per_gpu": total_mb * grad_per_param / num_gpus,
        "optimizer_per_gpu": total_mb * adam_states_per_param / num_gpus,
        "total_per_gpu": total_mb * (1 + grad_per_param + adam_states_per_param) / num_gpus,
    }

    return stages


def print_memory_table(stages, model_size_million, num_gpus):
    print(f"\n=== ZeRO Memory Usage (Model: {model_size_million}M params, {num_gpus} GPUs) ===")
    header = f"{'Stage':<12} {'Params (MB)':<14} {'Grads (MB)':<12} {'Optim (MB)':<12} {'Total (MB)':<12}"
    print(header)
    print("-" * 64)
    for stage, mem in stages.items():
        print(
            f"{stage:<12} "
            f"{mem['params_per_gpu']:>12.1f}  "
            f"{mem['grads_per_gpu']:>10.1f}  "
            f"{mem['optimizer_per_gpu']:>10.1f}  "
            f"{mem['total_per_gpu']:>10.1f}"
        )

    print("\n=== Distributed Training Strategy Comparison ===")
    strategies = [
        ("Data Parallel (DDP)", "Model replicated on each GPU; gradients averaged", "Linear with GPUs", "Limited by single GPU memory"),
        ("ZeRO-1", "Optimizer states partitioned across GPUs", "Linear with GPUs", "Reduces optimizer memory by Nx"),
        ("ZeRO-2", "Optimizer + gradient states partitioned", "Linear with GPUs", "Further reduces memory"),
        ("ZeRO-3", "All states including model params partitioned", "Near-linear", "Maximum memory reduction (Nx total)"),
        ("Pipeline Parallel", "Model layers split across GPUs", "Moderate", "Reduces memory but adds pipeline bubble"),
        ("Tensor Parallel", "Individual layer operations split across GPUs", "Moderate", "Best for very large layers"),
    ]
    print(f"{'Strategy':<24} {'Description':<44} {'Scaling':<20} {'Memory Impact'}")
    print("-" * 110)
    for name, desc, scaling, memory in strategies:
        print(f"{name:<24} {desc:<44} {scaling:<20} {memory}")


def create_memory_chart(stages, model_size_million, num_gpus, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        stage_names = list(stages.keys())
        params = [stages[s]["params_per_gpu"] for s in stage_names]
        grads = [stages[s]["grads_per_gpu"] for s in stage_names]
        optim = [stages[s]["optimizer_per_gpu"] for s in stage_names]

        x = np.arange(len(stage_names))
        width = 0.5

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_params = ax.bar(x, params, width, label="Model Parameters", color="#4C72B0")
        bars_grads = ax.bar(x, grads, width, bottom=params, label="Gradients", color="#DD8452")
        bars_optim = ax.bar(x, optim, width, bottom=[p+g for p, g in zip(params, grads)], label="Optimizer States", color="#55A868")

        for i, (p, g, o) in enumerate(zip(params, grads, optim)):
            total = p + g + o
            ax.text(i, total + max(params)*0.01, f"{total:.0f}MB", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xlabel("ZeRO Stage", fontsize=12)
        ax.set_ylabel("Memory per GPU (MB)", fontsize=12)
        ax.set_title(f"ZeRO Stage Memory Comparison\n(Model: {model_size_million}M params, {num_gpus} GPUs, FP32)", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_names, fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(p+g+o for p, g, o in zip(params, grads, optim)) * 1.15)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nMemory comparison chart saved to {out_path}")
        return True
    except ImportError:
        print("matplotlib not available; skipping chart generation.")
        return False


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    zero2_path = os.path.join(out_dir, "deepspeed_zero2_config.json")
    zero3_path = os.path.join(out_dir, "deepspeed_zero3_config.json")

    with open(zero2_path, "w") as f:
        json.dump(DEEPSPEED_ZERO2_CONFIG, f, indent=2)
    print(f"Saved ZeRO-2 config: {zero2_path}")

    with open(zero3_path, "w") as f:
        json.dump(DEEPSPEED_ZERO3_CONFIG, f, indent=2)
    print(f"Saved ZeRO-3 config: {zero3_path}")

    total_params = data_parallel_training_demo()

    model_size_million = total_params / 1e6
    num_gpus = 8
    stages = zero_stage_memory_analysis(model_size_million, num_gpus=num_gpus)
    print_memory_table(stages, model_size_million, num_gpus)

    print(f"\n=== ZeRO Parameter Count Partitioning ===")
    for stage_name, mem in stages.items():
        param_fraction = mem["params_per_gpu"] / (model_size_million * 1e6 * 4 / (1024**2))
        optim_fraction = mem["optimizer_per_gpu"] / ((model_size_million * 1e6 * 4 * 3) / (1024**2))
        grad_fraction = mem["grads_per_gpu"] / (model_size_million * 1e6 * 4 / (1024**2))
        print(f"  {stage_name}: params={param_fraction:.2%}/GPU, grads={grad_fraction:.2%}/GPU, optim={optim_fraction:.2%}/GPU")

    chart_path = os.path.join(out_dir, "memory_comparison.png")
    create_memory_chart(stages, model_size_million, num_gpus, chart_path)


if __name__ == "__main__":
    main()
