import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {DEVICE}\n")


def demo_gpt2():
    print("=" * 60)
    print("GPT-2: Text Generation")
    print("=" * 60)

    gen = pipeline("text-generation", model="gpt2", device=DEVICE)
    prompt = "The future of artificial intelligence is"
    outputs = gen(prompt, max_new_tokens=40, num_return_sequences=2, do_sample=True, temperature=0.9)
    for i, o in enumerate(outputs):
        print(f"[{i+1}] {o['generated_text']}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer(prompt, return_tensors="pt")
    print(f"\ntokenized ids: {tokens['input_ids']}")
    print(f"decoded tokens: {[tokenizer.decode([t]) for t in tokens['input_ids'][0]]}")

    model = AutoModel.from_pretrained("gpt2").to(DEVICE)
    with torch.no_grad():
        out = model(**{k: v.to(DEVICE) for k, v in tokens.items()})
    print(f"last_hidden_state shape: {out.last_hidden_state.shape}")
    print()


def demo_bert_mlm():
    print("=" * 60)
    print("BERT: Masked LM (Japanese)")
    print("=" * 60)

    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)

    text = "東京は日本の[MASK]です。"
    inputs = tokenizer(text, return_tensors="pt")
    print(f"input text: {text}")
    print(f"token ids: {inputs['input_ids']}")
    print(f"tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

    mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        logits = model(**{k: v.to(DEVICE) for k, v in inputs.items()}).logits

    top_k = logits[0, mask_index, :].topk(5)
    print("\nTop-5 predictions for [MASK]:")
    for score, idx in zip(top_k.values[0], top_k.indices[0]):
        token = tokenizer.decode([idx.item()])
        print(f"  {token!r:12s}  score={score.item():.4f}")

    fill = pipeline("fill-mask", model=model_name, device=DEVICE)
    results = fill("日本語の[MASK]処理は難しい。")
    print("\nfill-mask pipeline results:")
    for r in results[:3]:
        print(f"  score={r['score']:.4f}  token={r['token_str']!r:10s}  seq={r['sequence']}")
    print()


def demo_mt5():
    print("=" * 60)
    print("mT5: Text-to-Text Generation")
    print("=" * 60)

    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    prompts = [
        "translate English to French: Hello, how are you?",
        "summarize: The cat sat on the mat and looked at the dog.",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        print(f"input: {prompt}")
        print(f"encoder input shape: {inputs['input_ids'].shape}")

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=4,
                early_stopping=True,
            )
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"output: {decoded}\n")

    t2t = pipeline("text2text-generation", model=model_name, device=DEVICE)
    result = t2t("translate English to German: Good morning.", max_new_tokens=30)
    print(f"pipeline text2text: {result[0]['generated_text']}")
    print()


if __name__ == "__main__":
    demo_gpt2()
    demo_bert_mlm()
    demo_mt5()
