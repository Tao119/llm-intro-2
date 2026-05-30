"""
rag_reranking.py

Two-stage RAG retrieval experiment:
  Stage 1: BM25-style TF-IDF retrieval (pure Python) — top-10
  Stage 2: BERT cross-encoder reranking (CLS cosine) — reorder top-10

Compares BM25-only vs BM25+reranking on 10 test queries
using Precision@3 as the metric.

Knowledge base: 30 Japanese documents (same KNOWLEDGE_BASE as ch13).
"""

import os
import math
import json
import collections
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

OUT_DIR = os.path.dirname(__file__)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Knowledge base (30 documents)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    ("東京の概要",     "東京は日本の首都であり、世界最大級の都市圏を持つ。人口は約1400万人で、政治・経済・文化の中心地として機能している。23の特別区と多摩地域から構成される。"),
    ("東京の観光",     "東京には浅草の雷門、東京スカイツリー、上野動物園、東京ディズニーランドなど多くの観光スポットがある。渋谷のスクランブル交差点は世界的に有名。"),
    ("富士山",        "富士山は静岡県と山梨県にまたがる標高3776メートルの日本最高峰。2013年にユネスコ世界文化遺産に登録。毎年夏には約30万人が登山する。"),
    ("新幹線の歴史",   "新幹線は1964年の東京オリンピック開催に合わせて開業した世界初の高速鉄道。東海道新幹線（東京〜大阪）が最初の路線で、最高速度は現在320km/h。"),
    ("日本の食文化",   "日本食はユネスコ無形文化遺産に登録されている。寿司、天ぷら、ラーメン、うどんなどが代表的。日本には世界最多のミシュラン星付きレストランがある。"),
    ("京都の文化",    "京都は794年から1869年まで日本の首都だった古都。金閣寺、清水寺、嵐山、伏見稲荷大社など多数の世界遺産を有し、年間5000万人以上の観光客が訪れる。"),
    ("日本の気候",    "日本は温帯モンスーン気候に属し、四季が明確。梅雨は6〜7月に発生し、台風シーズンは9〜10月。北海道は寒冷、沖縄は亜熱帯性気候と南北で大きく異なる。"),
    ("日本の経済",    "日本はGDP世界第3位の経済大国。製造業（自動車・電機）が主要産業で、トヨタ、ソニー、任天堂などの世界的企業が本社を置く。近年はサービス業の比重も増大。"),
    ("桜の文化",      "桜は日本の国花的存在で、毎年3〜5月に開花。花見は平安時代から続く伝統文化。気象庁が桜前線を発表し、全国で花見イベントが開催される。"),
    ("大阪の特徴",    "大阪は日本第3の都市で、商業・文化の中心地。「天下の台所」とも呼ばれる食の街で、たこ焼き・お好み焼きが名物。道頓堀は世界的な観光スポット。"),
    ("北海道の魅力",  "北海道は日本最北の島で面積は日本の22%。広大な農地での農業・酪農が盛んで、ジャガイモ・乳製品が有名。冬の雪まつりと夏のラベンダー畑が人気。"),
    ("沖縄の文化",    "沖縄は琉球王国の歴史を持つ独自文化の地。首里城（世界遺産）、美しいサンゴ礁、三線音楽が特徴。沖縄料理（ゴーヤーチャンプルーなど）は長寿食として有名。"),
    ("日本の教育",    "日本の教育制度は6・3・3・4制で、義務教育は9年間。識字率はほぼ100%で世界最高水準。東京大学・京都大学などが有力な研究機関として知られる。"),
    ("日本の伝統文化", "日本の伝統文化には歌舞伎、能、茶道、花道、相撲などがある。多くがユネスコ無形文化遺産に登録されており、現代でも継承されている。"),
    ("日本の技術",    "日本は半導体・ロボット・精密機器などの先端技術で世界をリードする。自動化技術の分野では産業用ロボットの生産・輸出で世界首位を維持している。"),
    ("日本の宗教",    "日本では神道と仏教が共存する独特の宗教文化がある。初詣は神社で行い、葬式は仏教式というように、両者が生活に溶け込んでいる。"),
    ("日本の人口問題", "日本は少子高齢化が深刻で、合計特殊出生率は1.2程度。2008年をピークに人口減少が始まり、2060年には8700万人程度まで減少すると予測されている。"),
    ("アニメ・マンガ文化", "日本のアニメ・マンガは世界的なソフトパワーとなっている。スタジオジブリ、ドラゴンボール、ワンピースなどが世界中でファンを持ち、クールジャパン政策を支える。"),
    ("日本の交通",    "日本の交通網は世界最高水準。新幹線網が全国を結び、都市部の地下鉄・私鉄は時刻通りの運行で有名。成田・羽田空港は国際交通の要衝。"),
    ("日本の自然災害", "日本は地震・台風・火山噴火など自然災害が多い国。環太平洋火山帯に位置し、年間約1500回の地震が発生。2011年の東日本大震災は甚大な被害をもたらした。"),
    ("武士道と侍文化", "武士は平安時代末期から江戸時代まで日本の支配階層。武士道精神（誠実・名誉・忠義）は現代の日本文化にも影響を与え続けている。"),
    ("日本語の特徴",  "日本語は世界で最も複雑な言語のひとつ。ひらがな・カタカナ・漢字の3種の文字を使用し、敬語体系が発達している。話者数は約1億3000万人。"),
    ("温泉文化",      "日本全国に約3000ヶ所の温泉地がある。別府温泉（大分）・草津温泉（群馬）・有馬温泉（兵庫）が三大名湯。日本人の温泉文化は入浴そのものを癒しとして重視する。"),
    ("日本の農業",    "日本の農業は高品質で知られ、品種改良が発達。コメが基幹作物で、魚沼産コシヒカリなどのブランド米が有名。農業の担い手不足・後継者問題が深刻化している。"),
    ("デジタル化推進", "日本政府はデジタル庁を設置し行政のデジタル化を推進。マイナンバーカード普及、電子政府サービスの拡充、DX（デジタルトランスフォーメーション）促進に取り組んでいる。"),
    ("日本の医療",    "日本は国民皆保険制度を採用し、全国民が医療保険に加入。平均寿命は男性81歳・女性87歳で世界最長クラス。医療技術・設備は世界トップレベルを誇る。"),
    ("環境政策",      "日本は2050年カーボンニュートラルを目標として掲げる。太陽光・風力などの再生可能エネルギー拡大、電気自動車普及促進、省エネ技術の開発に注力している。"),
    ("伝統工芸",      "日本には西陣織・有田焼・南部鉄器など多数の伝統工芸品がある。経済産業省が230品目以上を伝統的工芸品として指定し保護・振興を図っている。"),
    ("ゲーム産業",    "日本のゲーム産業は世界をリードする規模を誇る。任天堂・ソニー・カプコンなどが世界的ブランドを展開し、マリオ・ポケモン・ファイナルファンタジーが世界的人気を誇る。"),
    ("日本の祭り文化", "日本全国で数万の祭りが年間を通じて開催される。京都の祇園祭・青森のねぶた祭・徳島の阿波踊りが三大祭りとして有名。地域コミュニティの結束を深める役割がある。"),
]

# 10 test queries: (query_text, correct_document_title)
TEST_QUERIES = [
    ("東京はどのような都市ですか？",         "東京の概要"),
    ("新幹線はいつ開業しましたか？",         "新幹線の歴史"),
    ("日本の食文化の特徴を教えてください。",  "日本の食文化"),
    ("富士山の高さはどのくらいですか？",      "富士山"),
    ("京都には何年まで首都がありましたか？",   "京都の文化"),
    ("日本のアニメはどのくらい人気がありますか？", "アニメ・マンガ文化"),
    ("日本の人口問題について教えてください。",  "日本の人口問題"),
    ("温泉地として有名な場所はどこですか？",   "温泉文化"),
    ("日本の伝統的な文化について教えてください。", "日本の伝統文化"),
    ("大阪の名物料理は何ですか？",          "大阪の特徴"),
]


# ---------------------------------------------------------------------------
# BM25-style TF-IDF retrieval (pure Python, character n-gram tokenisation)
# ---------------------------------------------------------------------------

def char_ngram_tokenize(text, n=2):
    """Character bigram tokeniser — no external library required."""
    tokens = []
    for i in range(len(text) - n + 1):
        tokens.append(text[i:i + n])
    # Also add unigrams for very short texts
    tokens += list(text)
    return tokens


def build_tfidf_index(documents):
    """Build TF and IDF tables for BM25-style scoring."""
    tokenized = [char_ngram_tokenize(doc) for doc in documents]
    n_docs    = len(documents)

    # Document frequency
    df = collections.Counter()
    for tokens in tokenized:
        for t in set(tokens):
            df[t] += 1

    # IDF (smoothed)
    idf = {t: math.log((n_docs + 1) / (df[t] + 1)) + 1.0 for t in df}

    # TF-IDF vectors as dicts
    doc_vectors = []
    for tokens in tokenized:
        tf = collections.Counter(tokens)
        total = sum(tf.values())
        vec = {}
        for t, cnt in tf.items():
            vec[t] = (cnt / total) * idf.get(t, 0.0)
        doc_vectors.append(vec)

    return doc_vectors, idf


def query_score(query_tokens, doc_vec, idf):
    """Cosine-like dot product between query and document vector."""
    score = 0.0
    tf_q  = collections.Counter(query_tokens)
    total = sum(tf_q.values())
    for t, cnt in tf_q.items():
        q_weight = (cnt / total) * idf.get(t, 0.0)
        score   += q_weight * doc_vec.get(t, 0.0)
    return score


def bm25_retrieve(query, doc_vectors, idf, k=10):
    q_tokens = char_ngram_tokenize(query)
    scores   = [query_score(q_tokens, dv, idf) for dv in doc_vectors]
    ranked   = sorted(range(len(scores)), key=lambda i: -scores[i])
    return ranked[:k], [scores[i] for i in ranked[:k]]


# ---------------------------------------------------------------------------
# BERT cross-encoder reranking (CLS cosine similarity)
# ---------------------------------------------------------------------------

def load_bert_model(model_name):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def bert_encode(tokenizer, model, text, device, max_length=256):
    import torch
    enc = tokenizer(text, max_length=max_length, truncation=True,
                    padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
    return cls[0]


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rerank_with_bert(query, candidate_indices, documents, tokenizer, model, device):
    import torch
    q_vec = bert_encode(tokenizer, model, query, device)
    scores = []
    for idx in candidate_indices:
        d_vec = bert_encode(tokenizer, model, documents[idx], device)
        scores.append(cosine_sim(q_vec, d_vec))
    order = sorted(range(len(candidate_indices)), key=lambda i: -scores[i])
    return [candidate_indices[i] for i in order], [scores[i] for i in order]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def precision_at_k(ranked_indices, correct_title, titles, k=3):
    top_k_titles = {titles[i] for i in ranked_indices[:k]}
    return 1.0 if correct_title in top_k_titles else 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(bm25_scores, rerank_scores, query_labels, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        cjk_font = None
        for fp in fm.findSystemFonts():
            name = os.path.basename(fp).lower()
            if any(k in name for k in ("noto", "gothic", "hiragino", "meiryo", "ipag")):
                try:
                    prop = fm.FontProperties(fname=fp)
                    cjk_font = prop
                    break
                except Exception:
                    continue

        x = np.arange(len(query_labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width / 2, bm25_scores, width, label="BM25 only", color="steelblue", alpha=0.8)
        bars2 = ax.bar(x + width / 2, rerank_scores, width, label="BM25 + BERT reranking", color="coral", alpha=0.8)

        ax.set_xlabel("Query index")
        ax.set_ylabel("Precision@3 (1=hit, 0=miss)")
        ax.set_title("BM25 vs BM25+BERT Reranking — Precision@3")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(query_labels))], fontsize=9)
        ax.legend()
        ax.set_ylim(-0.1, 1.3)
        ax.axhline(np.mean(bm25_scores), color="steelblue", linestyle="--", alpha=0.5,
                   label=f"BM25 mean P@3={np.mean(bm25_scores):.2f}")
        ax.axhline(np.mean(rerank_scores), color="coral", linestyle="--", alpha=0.5,
                   label=f"Rerank mean P@3={np.mean(rerank_scores):.2f}")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved: {out_path}")
    except Exception as e:
        print(f"Plot skipped ({e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    titles    = [t for t, _ in KNOWLEDGE_BASE]
    documents = [d for _, d in KNOWLEDGE_BASE]

    print(f"Knowledge base: {len(documents)} documents")
    print("Building TF-IDF index...")
    doc_vectors, idf = build_tfidf_index(documents)
    print("TF-IDF index ready.\n")

    # Load BERT model
    bert_model_name = "cl-tohoku/bert-base-japanese-v3"
    try:
        print(f"Loading BERT model: {bert_model_name}")
        bert_tokenizer, bert_model = load_bert_model(bert_model_name)
        bert_model = bert_model.to(device)
        bert_model.eval()
        bert_available = True
        print("BERT model loaded.\n")
    except Exception as e:
        print(f"BERT unavailable ({e}), falling back to multilingual BERT\n")
        try:
            bert_model_name = "bert-base-multilingual-cased"
            bert_tokenizer, bert_model = load_bert_model(bert_model_name)
            bert_model = bert_model.to(device)
            bert_model.eval()
            bert_available = True
            print(f"Fallback BERT loaded: {bert_model_name}\n")
        except Exception as e2:
            print(f"BERT completely unavailable ({e2}), reranking will use BM25 scores.\n")
            bert_available = False

    bm25_p3_scores   = []
    rerank_p3_scores = []
    all_results      = []

    print("=" * 70)
    print(f"{'Q':>2}  {'BM25@3':>7}  {'Rerank@3':>9}  Query")
    print("-" * 70)

    for qi, (query, correct_title) in enumerate(TEST_QUERIES):
        # Stage 1: BM25 top-10
        bm25_ranked, bm25_scores_vals = bm25_retrieve(query, doc_vectors, idf, k=10)

        # Stage 2: BERT reranking
        if bert_available:
            try:
                reranked, rerank_scores_vals = rerank_with_bert(
                    query, bm25_ranked, documents, bert_tokenizer, bert_model, device
                )
            except Exception as e:
                print(f"  Reranking failed for Q{qi+1}: {e}")
                reranked = bm25_ranked
                rerank_scores_vals = bm25_scores_vals
        else:
            reranked = bm25_ranked
            rerank_scores_vals = bm25_scores_vals

        bm25_p3   = precision_at_k(bm25_ranked, correct_title, titles, k=3)
        rerank_p3 = precision_at_k(reranked, correct_title, titles, k=3)

        bm25_p3_scores.append(bm25_p3)
        rerank_p3_scores.append(rerank_p3)

        print(f"{qi+1:>2}  {bm25_p3:>7.0f}  {rerank_p3:>9.0f}  {query[:30]}")

        all_results.append({
            "query": query,
            "correct_title": correct_title,
            "bm25_top3": [titles[i] for i in bm25_ranked[:3]],
            "reranked_top3": [titles[i] for i in reranked[:3]],
            "bm25_precision_at_3": bm25_p3,
            "rerank_precision_at_3": rerank_p3,
        })

    print("=" * 70)
    mean_bm25   = np.mean(bm25_p3_scores)
    mean_rerank = np.mean(rerank_p3_scores)
    print(f"\nMean Precision@3  BM25: {mean_bm25:.3f}   BM25+BERT: {mean_rerank:.3f}")
    delta = mean_rerank - mean_bm25
    direction = "improvement" if delta >= 0 else "regression"
    print(f"Delta: {delta:+.3f} ({direction})\n")

    # Plot
    plot_path = os.path.join(OUT_DIR, "retrieval_comparison.png")
    plot_comparison(bm25_p3_scores, rerank_p3_scores,
                    [f"Q{i+1}" for i in range(len(TEST_QUERIES))], plot_path)

    # Save results
    summary = {
        "bert_model": bert_model_name if bert_available else "unavailable",
        "n_documents": len(documents),
        "n_queries": len(TEST_QUERIES),
        "bm25_mean_precision_at_3": float(mean_bm25),
        "rerank_mean_precision_at_3": float(mean_rerank),
        "delta": float(delta),
        "per_query": all_results,
    }
    out_path = os.path.join(OUT_DIR, "rag_reranking_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
