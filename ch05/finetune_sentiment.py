import os
import sys
import numpy as np

sys.path.append("..")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "marc-ja")
MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
MAX_LEN = 128


def load_marc_ja():
    ds = load_dataset("shunk031/JGLUE", name="MARC-ja", trust_remote_code=True)
    print(f"train size: {len(ds['train'])}, validation size: {len(ds['validation'])}")
    print(f"sample: {ds['train'][0]}")
    return ds


def build_tokenize_fn(tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LEN,
        )
    return tokenize


def encode_labels(example):
    label_map = {"negative": 0, "positive": 1}
    return {"labels": label_map[example["label"]]}


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_marc_ja()

    ds = ds.map(encode_labels)
    ds = ds.map(build_tokenize_fn(tokenizer), batched=True, remove_columns=["sentence", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"\nfinal accuracy: {metrics['eval_accuracy']:.4f}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
