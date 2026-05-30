# 大規模言語モデル入門II 生成型LLMの実装と評価

「大規模言語モデル入門」の続編。ch10〜ch14。  
Swallow-7B（東工大製、Llama2ベース日本語LLM）を軸に実装。  
GPU環境（NVIDIA A100/L4 or Apple MPS）推奨。

## 章構成

| 章 | テーマ | キーワード | 状態 |
|----|--------|-----------|------|
| ch10 | 性能評価 | llm-jp-eval, Japanese Vicuna QA, LLM-as-a-judge | 実装中 |
| ch11 | 指示チューニング | QLoRA, Swallow-7B, チャットテンプレート, FLAN形式 | 予定 |
| ch12 | 選好チューニング | RLHF, DPO, 報酬モデル | 予定 |
| ch13 | RAG | LangChain, BM25, DPR, LLMへの指示チューニング | 予定 |
| ch14 | 分散並列学習 | DeepSpeed ZeRO, Megatron-LM, 3次元並列化 | 予定 |

## 実行環境

```
torch >= 2.0
transformers >= 4.40
peft (QLoRA)
langchain
faiss-cpu / faiss-gpu
deepspeed (ch14のみ)
```

## ディレクトリ構成

```
llm-intro-2/
├── ch10/         # 性能評価スクリプト
├── ch11/         # 指示チューニング
├── ch12/         # 選好チューニング (DPO)
├── ch13/         # RAG構築
├── ch14/         # 分散並列学習設定
├── common/       # 共通ユーティリティ
└── experiments/  # 実験レポート
```
