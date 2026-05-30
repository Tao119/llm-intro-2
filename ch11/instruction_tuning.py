import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

INSTRUCTION_DATA = [
    {"instruction": "日本の首都を教えてください。", "input": "", "output": "日本の首都は東京です。"},
    {"instruction": "以下の文章を要約してください。", "input": "東京は日本最大の都市で、人口は約1400万人です。多くの企業や文化施設が集まっており、世界有数の経済都市です。", "output": "東京は人口約1400万人の日本最大の都市で、経済・文化の中心地です。"},
    {"instruction": "次の計算をしてください。", "input": "15 + 27 = ?", "output": "15 + 27 = 42です。"},
    {"instruction": "「猫」を英語に翻訳してください。", "input": "", "output": "「猫」の英語は「cat」です。"},
    {"instruction": "桜の花が咲く季節はいつですか？", "input": "", "output": "桜の花は主に春（3月から5月）に咲きます。"},
    {"instruction": "以下の文章の誤字を修正してください。", "input": "きょうは天気がよくて、さんぽに出かけたら気持ちよかたです。", "output": "今日は天気がよくて、散歩に出かけたら気持ちよかったです。"},
    {"instruction": "富士山の高さを教えてください。", "input": "", "output": "富士山の高さは3776メートルです。日本最高峰の山です。"},
    {"instruction": "水の化学式は何ですか？", "input": "", "output": "水の化学式はH₂O（H2O）です。水素原子2つと酸素原子1つからできています。"},
    {"instruction": "以下の単語を使って文を作ってください。", "input": "犬、公園、走る", "output": "犬が公園を元気よく走っています。"},
    {"instruction": "江戸幕府を開いたのは誰ですか？", "input": "", "output": "江戸幕府を開いたのは徳川家康です。1603年に征夷大将軍に任命されました。"},
    {"instruction": "以下の文章を丁寧な言葉に直してください。", "input": "明日来て。", "output": "明日お越しいただけますでしょうか。"},
    {"instruction": "円周率πの近似値を教えてください。", "input": "", "output": "円周率πの近似値は約3.14159です。"},
    {"instruction": "「ありがとう」を韓国語で何と言いますか？", "input": "", "output": "「ありがとう」は韓国語で「감사합니다（カムサハムニダ）」と言います。"},
    {"instruction": "以下の数列の次の数を答えてください。", "input": "1, 1, 2, 3, 5, 8, 13, ?", "output": "次の数は21です。これはフィボナッチ数列で、前の2つの数の和が次の数になります。"},
    {"instruction": "日本で一番長い川はどこですか？", "input": "", "output": "日本で一番長い川は信濃川で、全長367キロメートルです。"},
    {"instruction": "以下のレシピを簡単に説明してください。", "input": "材料：卵2個、塩少々、バター10g。作り方：フライパンにバターを溶かし、溶き卵を流し込み、塩で味を整えて炒める。", "output": "バターで溶き卵を炒め、塩で味付けするシンプルなスクランブルエッグのレシピです。"},
    {"instruction": "光の速さはどのくらいですか？", "input": "", "output": "光の速さは真空中で毎秒約30万キロメートル（299,792,458 m/s）です。"},
    {"instruction": "以下の文章のキーワードを3つ抽出してください。", "input": "人工知能技術の急速な発展により、自然言語処理や画像認識の分野で革命的な変化が起きています。", "output": "キーワード：人工知能、自然言語処理、画像認識"},
    {"instruction": "地球の人口はおよそ何人ですか？", "input": "", "output": "地球の人口は2024年時点でおよそ81億人です。"},
    {"instruction": "以下の文を否定文に変えてください。", "input": "今日は晴れています。", "output": "今日は晴れていません。"},
    {"instruction": "DNAとは何ですか？", "input": "", "output": "DNAとはデオキシリボ核酸の略で、生物の遺伝情報を担う分子です。二重らせん構造を持ちます。"},
    {"instruction": "以下の詩を読んで感想を述べてください。", "input": "古池や 蛙飛びこむ 水の音", "output": "松尾芭蕉の有名な俳句です。静寂な古池に蛙が飛び込む瞬間の音を捉え、自然の静けさと生命の動きが対比された美しい作品です。"},
    {"instruction": "東京から大阪まで新幹線でどのくらいかかりますか？", "input": "", "output": "東京から大阪まで東海道新幹線（のぞみ）で約2時間30分かかります。"},
    {"instruction": "以下の英文を日本語に翻訳してください。", "input": "The quick brown fox jumps over the lazy dog.", "output": "素早い茶色のキツネが怠け者の犬を飛び越えました。"},
    {"instruction": "ビタミンCが多く含まれる食品を3つ挙げてください。", "input": "", "output": "ビタミンCが多く含まれる食品は、レモン、ピーマン、ブロッコリーです。"},
    {"instruction": "以下の数学の問題を解いてください。", "input": "二次方程式 x² - 5x + 6 = 0 を解いてください。", "output": "x² - 5x + 6 = 0 を因数分解すると (x-2)(x-3) = 0 となり、x = 2 または x = 3 です。"},
    {"instruction": "俳句を一句作ってください。", "input": "テーマ：夏", "output": "夏空や 雲ひとつなく 蝉時雨"},
    {"instruction": "以下の文章の主語と述語を教えてください。", "input": "美しい桜の花が風に揺れています。", "output": "主語：桜の花が、述語：揺れています"},
    {"instruction": "プログラミング言語Pythonの特徴を教えてください。", "input": "", "output": "Pythonはシンプルな文法、豊富なライブラリ、データサイエンスやAI分野での活用が特徴の汎用プログラミング言語です。"},
    {"instruction": "以下の物語の続きを書いてください。", "input": "ある日、太郎は森の中で不思議な小屋を見つけました。", "output": "ドアをそっと開けると、中には本がぎっしり並んでいました。棚の奥から光が漏れ、太郎は恐る恐る近づいていきました。"},
    {"instruction": "地球温暖化の主な原因は何ですか？", "input": "", "output": "地球温暖化の主な原因は、化石燃料の燃焼による二酸化炭素などの温室効果ガスの増加です。"},
    {"instruction": "以下の単語の反対語を答えてください。", "input": "明るい", "output": "「明るい」の反対語は「暗い」です。"},
    {"instruction": "日本の国旗について説明してください。", "input": "", "output": "日本の国旗は「日の丸」と呼ばれ、白地の中央に赤い円が描かれています。赤い円は太陽を表しています。"},
    {"instruction": "以下の文章から数字を全て抽出してください。", "input": "今年は2024年で、3月に桜が咲き、気温は25度になりました。", "output": "抽出された数字：2024、3、25"},
    {"instruction": "睡眠の重要性について説明してください。", "input": "", "output": "睡眠は体の回復、記憶の定着、免疫機能の維持に重要です。成人には7〜9時間の睡眠が推奨されています。"},
    {"instruction": "以下の文章の時制を過去形に変えてください。", "input": "花が咲いています。", "output": "花が咲いていました。"},
    {"instruction": "インターネットの歴史を簡単に説明してください。", "input": "", "output": "インターネットは1960年代の米国軍の研究から始まり、1990年代にWWWの発明で一般普及し、現在では世界50億人以上が利用しています。"},
    {"instruction": "以下の料理の手順を番号付きで整理してください。", "input": "水を沸かして、麺を入れて、茹で上がったらスープを加えて、トッピングをのせる。", "output": "1. 水を沸かす\n2. 麺を入れる\n3. 茹で上がったらスープを加える\n4. トッピングをのせる"},
    {"instruction": "月はなぜ地球の周りを回っているのですか？", "input": "", "output": "月は地球の重力に引き付けられており、その引力と月の公転速度のバランスによって地球の周りを回り続けています。"},
    {"instruction": "以下の文章をより簡潔に書き直してください。", "input": "私は今日の朝、起きてから朝食を食べて、その後で家を出て学校に向かって歩いていきました。", "output": "今朝、朝食を済ませてから徒歩で登校しました。"},
    {"instruction": "太陽系の惑星を全て挙げてください。", "input": "", "output": "太陽系の惑星は、水星、金星、地球、火星、木星、土星、天王星、海王星の8つです。"},
    {"instruction": "敬語の種類を教えてください。", "input": "", "output": "敬語には尊敬語（相手の行為を高める）、謙譲語（自分の行為を下げる）、丁寧語（丁寧に表現する）の3種類があります。"},
    {"instruction": "以下の慣用句の意味を説明してください。", "input": "猫の手も借りたい", "output": "「猫の手も借りたい」とは、非常に忙しくて猫のような役に立たないものでも手伝ってほしいというほど、人手が足りない状況を表す慣用句です。"},
    {"instruction": "電子書籍と紙の本の違いを教えてください。", "input": "", "output": "電子書籍は持ち運びやすく大量保存できますが、目が疲れやすい面があります。紙の本は読みやすく所有感がありますが、かさばります。"},
    {"instruction": "プロバイオティクスとは何ですか？", "input": "", "output": "プロバイオティクスとは、腸内環境を改善し健康に良い影響を与える生きた微生物のことです。ヨーグルトや発酵食品に多く含まれています。"},
    {"instruction": "以下の英単語の意味を教えてください。", "input": "serendipity", "output": "「serendipity」は偶然の幸運な発見や、思いがけない幸せな出来事という意味の英単語です。"},
    {"instruction": "日本の伝統的な祭りを3つ教えてください。", "input": "", "output": "日本の伝統的な祭りは、京都の祇園祭、青森のねぶた祭、徳島の阿波踊りが有名です。"},
    {"instruction": "ストレス解消法を3つ提案してください。", "input": "", "output": "ストレス解消法として、適度な運動、趣味の時間を持つこと、十分な睡眠の3つをお勧めします。"},
    {"instruction": "以下の状況に適した謝罪の言葉を教えてください。", "input": "仕事で大きなミスをしてしまった。", "output": "「この度は大変なご迷惑をおかけし、深くお詫び申し上げます。今後このようなことがないよう、再発防止に努めてまいります。」"},
    {"instruction": "量子コンピュータとは何ですか？簡単に説明してください。", "input": "", "output": "量子コンピュータは量子力学の原理を利用した次世代コンピュータです。通常のビットではなく量子ビット（キュービット）を使用し、特定の計算を従来型より飛躍的に高速に処理できます。"},
]

ALPACA_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
ALPACA_TEMPLATE_NO_INPUT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"


def format_alpaca(item):
    if item["input"]:
        return ALPACA_TEMPLATE.format(**item)
    return ALPACA_TEMPLATE_NO_INPUT.format(**item)


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.encodings = []
        for item in data:
            text = format_alpaca(item)
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            self.encodings.append({
                "input_ids": ids,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": ids.clone(),
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def main():
    model_name = "rinna/japanese-gpt2-medium"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = InstructionDataset(INSTRUCTION_DATA, tokenizer)

    use_mps = torch.backends.mps.is_available()
    output_dir = os.path.join(os.path.dirname(__file__), "results")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        warmup_steps=20,
        save_strategy="epoch",
        logging_steps=5,
        no_cuda=not torch.cuda.is_available(),
        use_mps_device=use_mps,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nAdapter saved to {output_dir}")

    if use_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    test_instruction = {"instruction": "日本の首都を教えてください。", "input": "", "output": ""}
    prompt = ALPACA_TEMPLATE_NO_INPUT.format(
        instruction=test_instruction["instruction"],
        output="",
    ).rstrip()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n=== Sample Response ===")
    print(f"Instruction: {test_instruction['instruction']}")
    print(f"Response: {generated.strip()}")


if __name__ == "__main__":
    main()
