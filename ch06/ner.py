import os
import sys
import numpy as np

sys.path.append("..")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "ner")
MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
DATASET_NAME = "llm-book/ner-wikipedia-dataset"
MAX_LEN = 128


def build_label_mappings(ds):
    entity_types = set()
    for split in ds.values():
        for example in split:
            for tag in example["tags"]:
                if tag != "O":
                    entity_types.add(tag.split("-", 1)[1])

    labels = ["O"]
    for et in sorted(entity_types):
        labels.append(f"B-{et}")
        labels.append(f"I-{et}")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LEN,
        is_split_into_words=True,
    )
    all_labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word_id:
                aligned.append(label2id.get(tags[word_id], 0))
            else:
                tag = tags[word_id]
                if tag.startswith("B-"):
                    aligned.append(label2id.get("I-" + tag[2:], 0))
                else:
                    aligned.append(label2id.get(tag, 0))
            prev_word_id = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized


def build_compute_metrics(id2label):
    import evaluate
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels = [
            [id2label[l] for l in label_row if l != -100]
            for label_row in labels
        ]
        true_preds = [
            [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(preds, labels)
        ]

        result = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"],
            "accuracy": result["overall_accuracy"],
        }

    return compute_metrics


def main():
    ds = load_dataset(DATASET_NAME, trust_remote_code=True)
    print(f"splits: {list(ds.keys())}")
    print(f"train size: {len(ds['train'])}")
    print(f"sample tokens: {ds['train'][0]['tokens'][:10]}")
    print(f"sample tags:   {ds['train'][0]['tags'][:10]}")

    labels, label2id, id2label = build_label_mappings(ds)
    print(f"\nentity types: {sorted({l[2:] for l in labels if l != 'O'})}")
    print(f"total labels: {len(labels)}")
    print(f"label list: {labels}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        return tokenize_and_align_labels(examples, tokenizer, label2id)

    remove_cols = [c for c in ds["train"].column_names if c not in ("labels",)]
    tokenized_ds = ds.map(preprocess, batched=True, remove_columns=remove_cols)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
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
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = build_compute_metrics(id2label)

    eval_split = "validation" if "validation" in tokenized_ds else "test"

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds[eval_split],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"\nentity-level F1: {metrics['eval_f1']:.4f}")
    print(f"precision:       {metrics['eval_precision']:.4f}")
    print(f"recall:          {metrics['eval_recall']:.4f}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
