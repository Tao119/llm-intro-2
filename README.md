# 大規模言語モデル入門II 生成型LLMの実装と評価

「大規模言語モデル入門」の続編。ch10〜ch14。

## 章構成と実装内容

| 章 | テーマ | 実装 | 実行方法 |
|----|--------|------|---------|
| **ch10** | 性能評価 | Exact Match・Token F1・ROUGE-L・LLM-as-a-Judge プロンプト | `python3 ch10/evaluate.py` |
| **ch11** | 指示チューニング | Alpaca形式データセット + QLoRA (rinna/japanese-gpt2-medium) | `python3 ch11/instruction_tuning.py` |
| **ch12** | 選好チューニング | DPO損失fromスクラッチ + DPOTrainer クラス実装 | `python3 ch12/dpo.py` |
| **ch13** | RAG | Faiss類似検索 + LangChain統合 + RAGプロンプト + 生成デモ | `python3 ch13/rag_pipeline.py` |
| **ch14** | 分散並列学習 | DeepSpeed ZeRO設定JSON + DataParallelデモ + メモリ比較図 | `python3 ch14/distributed_training.py` |

## 環境構築

```bash
pip install transformers datasets peft faiss-cpu \
            langchain-community langchain-core sentencepiece
```

### ch11・ch12（7Bモデルを使う場合）
```bash
pip install bitsandbytes  # 4bit量子化
# GPU推奨（Swallow-7B等）。コード内のモデル名を変更して使用
```

## キーコンセプト

### DPO損失（ch12）
```
L_DPO = -E[log σ(β × (log π_θ(y+|x)/π_ref(y+|x) - log π_θ(y-|x)/π_ref(y-|x)))]
```
chosen(y+)とrejected(y-)のlogプロバビリティ差を最大化する。

### ZeROメモリ削減（ch14）

| Stage | Params | Grads | Optim | Total/GPU (8GPU) |
|-------|--------|-------|-------|-----------------|
| ZeRO-0 | 100% | 100% | 100% | 100% |
| ZeRO-1 | 100% | 100% | 12.5% | 約50% |
| ZeRO-2 | 100% | 12.5% | 12.5% | 約30% |
| ZeRO-3 | 12.5% | 12.5% | 12.5% | 約13% |

### RAGパイプライン（ch13）
```
質問 → 埋め込み → Faiss検索 → top-k文書取得
→ プロンプト構築: "以下を参考に回答: {context} 質問: {q}"
→ LLM生成
```
