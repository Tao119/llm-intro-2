import os
import json
import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

KNOWLEDGE_BASE = [
    ("東京の概要", "東京は日本の首都であり、世界最大級の都市圏を持つ。人口は約1400万人で、政治・経済・文化の中心地として機能している。23の特別区と多摩地域から構成される。"),
    ("東京の観光", "東京には浅草の雷門、東京スカイツリー、上野動物園、東京ディズニーランドなど多くの観光スポットがある。渋谷のスクランブル交差点は世界的に有名。"),
    ("富士山", "富士山は静岡県と山梨県にまたがる標高3776メートルの日本最高峰。2013年にユネスコ世界文化遺産に登録。毎年夏には約30万人が登山する。"),
    ("新幹線の歴史", "新幹線は1964年の東京オリンピック開催に合わせて開業した世界初の高速鉄道。東海道新幹線（東京〜大阪）が最初の路線で、最高速度は現在320km/h。"),
    ("日本の食文化", "日本食はユネスコ無形文化遺産に登録されている。寿司、天ぷら、ラーメン、うどんなどが代表的。日本には世界最多のミシュラン星付きレストランがある。"),
    ("京都の文化", "京都は794年から1869年まで日本の首都だった古都。金閣寺、清水寺、嵐山、伏見稲荷大社など多数の世界遺産を有し、年間5000万人以上の観光客が訪れる。"),
    ("日本の気候", "日本は温帯モンスーン気候に属し、四季が明確。梅雨は6〜7月に発生し、台風シーズンは9〜10月。北海道は寒冷、沖縄は亜熱帯性気候と南北で大きく異なる。"),
    ("日本の経済", "日本はGDP世界第3位の経済大国。製造業（自動車・電機）が主要産業で、トヨタ、ソニー、任天堂などの世界的企業が本社を置く。近年はサービス業の比重も増大。"),
    ("桜の文化", "桜は日本の国花的存在で、毎年3〜5月に開花。花見は平安時代から続く伝統文化。気象庁が桜前線を発表し、全国で花見イベントが開催される。"),
    ("大阪の特徴", "大阪は日本第3の都市で、商業・文化の中心地。「天下の台所」とも呼ばれる食の街で、たこ焼き・お好み焼きが名物。道頓堀は世界的な観光スポット。"),
    ("北海道の魅力", "北海道は日本最北の島で面積は日本の22%。広大な農地での農業・酪農が盛んで、ジャガイモ・乳製品が有名。冬の雪まつりと夏のラベンダー畑が人気。"),
    ("沖縄の文化", "沖縄は琉球王国の歴史を持つ独自文化の地。首里城（世界遺産）、美しいサンゴ礁、三線音楽が特徴。沖縄料理（ゴーヤーチャンプルーなど）は長寿食として有名。"),
    ("日本の教育", "日本の教育制度は6・3・3・4制で、義務教育は9年間。識字率はほぼ100%で世界最高水準。東京大学・京都大学などが有力な研究機関として知られる。"),
    ("日本の伝統文化", "日本の伝統文化には歌舞伎、能、茶道、花道、相撲などがある。多くがユネスコ無形文化遺産に登録されており、現代でも継承されている。"),
    ("日本の技術", "日本は半導体・ロボット・精密機器などの先端技術で世界をリードする。自動化技術の分野では産業用ロボットの生産・輸出で世界首位を維持している。"),
    ("日本の宗教", "日本では神道と仏教が共存する独特の宗教文化がある。初詣は神社で行い、葬式は仏教式というように、両者が生活に溶け込んでいる。"),
    ("日本の人口問題", "日本は少子高齢化が深刻で、合計特殊出生率は1.2程度。2008年をピークに人口減少が始まり、2060年には8700万人程度まで減少すると予測されている。"),
    ("アニメ・マンガ文化", "日本のアニメ・マンガは世界的なソフトパワーとなっている。スタジオジブリ、ドラゴンボール、ワンピースなどが世界中でファンを持ち、クールジャパン政策を支える。"),
    ("日本の交通", "日本の交通網は世界最高水準。新幹線網が全国を結び、都市部の地下鉄・私鉄は時刻通りの運行で有名。成田・羽田空港は国際交通の要衝。"),
    ("日本の自然災害", "日本は地震・台風・火山噴火など自然災害が多い国。環太平洋火山帯に位置し、年間約1500回の地震が発生。2011年の東日本大震災は甚大な被害をもたらした。"),
    ("武士道と侍文化", "武士は平安時代末期から江戸時代まで日本の支配階層。武士道精神（誠実・名誉・忠義）は現代の日本文化にも影響を与え続けている。"),
    ("日本語の特徴", "日本語は世界で最も複雑な言語のひとつ。ひらがな・カタカナ・漢字の3種の文字を使用し、敬語体系が発達している。話者数は約1億3000万人。"),
    ("温泉文化", "日本全国に約3000ヶ所の温泉地がある。別府温泉（大分）・草津温泉（群馬）・有馬温泉（兵庫）が三大名湯。日本人の温泉文化は入浴そのものを癒しとして重視する。"),
    ("日本の農業", "日本の農業は高品質で知られ、品種改良が発達。コメが基幹作物で、魚沼産コシヒカリなどのブランド米が有名。農業の担い手不足・後継者問題が深刻化している。"),
    ("デジタル化推進", "日本政府はデジタル庁を設置し行政のデジタル化を推進。マイナンバーカード普及、電子政府サービスの拡充、DX（デジタルトランスフォーメーション）促進に取り組んでいる。"),
    ("日本の医療", "日本は国民皆保険制度を採用し、全国民が医療保険に加入。平均寿命は男性81歳・女性87歳で世界最長クラス。医療技術・設備は世界トップレベルを誇る。"),
    ("環境政策", "日本は2050年カーボンニュートラルを目標として掲げる。太陽光・風力などの再生可能エネルギー拡大、電気自動車普及促進、省エネ技術の開発に注力している。"),
    ("伝統工芸", "日本には西陣織・有田焼・南部鉄器など多数の伝統工芸品がある。経済産業省が230品目以上を伝統的工芸品として指定し保護・振興を図っている。"),
    ("ゲーム産業", "日本のゲーム産業は世界をリードする規模を誇る。任天堂・ソニー・カプコンなどが世界的ブランドを展開し、マリオ・ポケモン・ファイナルファンタジーが世界的人気を誇る。"),
    ("日本の祭り文化", "日本全国で数万の祭りが年間を通じて開催される。京都の祇園祭・青森のねぶた祭・徳島の阿波踊りが三大祭りとして有名。地域コミュニティの結束を深める役割がある。"),
]

DEMO_QA = [
    ("東京はどのような都市ですか？", "東京"),
    ("新幹線はいつ開業しましたか？", "新幹線の歴史"),
    ("日本の食文化の特徴を教えてください。", "日本の食文化"),
]


def load_embedding_model(model_name):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def mean_pool(hidden, mask):
    mask_expanded = mask.unsqueeze(-1).float()
    return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


def embed_texts(model, tokenizer, texts, device, batch_size=8):
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=256, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def build_faiss_index(embeddings):
    import faiss
    dim = embeddings.shape[1]
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(dim)
    index.add(normed.astype(np.float32))
    return index, normed


def retrieve(query, model, tokenizer, index, device, k=3):
    q_emb = embed_texts(model, tokenizer, [query], device)
    q_normed = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    scores, indices = index.search(q_normed.astype(np.float32), k)
    return scores[0], indices[0]


def build_rag_prompt(query, retrieved_docs):
    context = "\n".join([f"[文書{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)])
    return f"以下の文書を参考に質問に答えてください。\n文書:{context}\n質問:{query}\n回答:"


def load_generator(model_name, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load generator {model_name}: {e}")
        return None, None


def generate_answer(prompt, gen_tokenizer, gen_model, device, max_new_tokens=80):
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=gen_tokenizer.eos_token_id,
        )
    generated = gen_tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    encoder_name = "cl-tohoku/bert-base-japanese-v3"
    try:
        enc_tokenizer, enc_model = load_embedding_model(encoder_name)
        enc_model = enc_model.to(device)
        print(f"Loaded encoder: {encoder_name}")
    except Exception as e:
        print(f"Failed to load {encoder_name}: {e}, falling back to multilingual BERT")
        encoder_name = "bert-base-multilingual-cased"
        enc_tokenizer, enc_model = load_embedding_model(encoder_name)
        enc_model = enc_model.to(device)

    kb_titles = [title for title, _ in KNOWLEDGE_BASE]
    kb_texts = [text for _, text in KNOWLEDGE_BASE]

    print(f"\nBuilding Faiss index for {len(kb_texts)} documents...")
    kb_embeddings = embed_texts(enc_model, enc_tokenizer, kb_texts, device)
    index, _ = build_faiss_index(kb_embeddings)
    print("Faiss index built.")

    try:
        from langchain_community.vectorstores import FAISS as LangFAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document
        print("\nUsing LangChain FAISS integration...")
        lc_embeddings = HuggingFaceEmbeddings(model_name=encoder_name)
        docs = [Document(page_content=text, metadata={"title": title}) for title, text in KNOWLEDGE_BASE]
        lc_index = LangFAISS.from_documents(docs, lc_embeddings)
        print("LangChain FAISS index created successfully.")
        langchain_available = True
    except Exception as e:
        print(f"LangChain integration skipped: {e}")
        langchain_available = False

    gen_model_name = "rinna/japanese-gpt2-medium"
    gen_tokenizer, gen_model = load_generator(gen_model_name, device)
    can_generate = gen_model is not None

    print("\n" + "="*60)
    print("RAG Pipeline Demo")
    print("="*60)

    retrieval_correct = 0
    results = []

    for query, expected_title in DEMO_QA:
        scores, indices = retrieve(query, enc_model, enc_tokenizer, index, device, k=3)
        retrieved_titles = [kb_titles[i] for i in indices]
        retrieved_texts = [kb_texts[i] for i in indices]

        prompt = build_rag_prompt(query, retrieved_texts)

        print(f"\nQuery: {query}")
        print("Retrieved documents:")
        for rank, (title, score) in enumerate(zip(retrieved_titles, scores), 1):
            hit = "[HIT]" if title == expected_title else "     "
            print(f"  {rank}. {hit} [{score:.4f}] {title}")

        if can_generate:
            answer = generate_answer(prompt, gen_tokenizer, gen_model, device)
            print(f"Generated answer: {answer}")
        else:
            print("[Generator unavailable — prompt ready for external LLM]")
            print(f"Prompt preview: {prompt[:200]}...")

        correct = expected_title in retrieved_titles
        if correct:
            retrieval_correct += 1

        results.append({
            "query": query,
            "expected_title": expected_title,
            "retrieved_titles": retrieved_titles,
            "correct_in_top3": correct,
        })

    precision = retrieval_correct / len(DEMO_QA)
    print(f"\n{'='*60}")
    print(f"Retrieval Precision@3: {precision:.4f} ({retrieval_correct}/{len(DEMO_QA)})")

    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, "rag_results.json"), "w", encoding="utf-8") as f:
        json.dump({"retrieval_precision_at_3": precision, "demos": results}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_dir}/rag_results.json")


if __name__ == "__main__":
    main()
