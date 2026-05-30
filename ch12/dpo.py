import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PREFERENCE_DATA = [
    {
        "prompt": "日本の首都はどこですか？",
        "chosen": "日本の首都は東京です。東京都は政治・経済・文化の中心地として機能しています。",
        "rejected": "わかりません。どこかの都市だと思います。",
    },
    {
        "prompt": "水の化学式を教えてください。",
        "chosen": "水の化学式はH₂Oです。水素原子2個と酸素原子1個から構成されています。",
        "rejected": "水は液体です。化学式は複雑なので覚えなくていいです。",
    },
    {
        "prompt": "富士山の高さはどのくらいですか？",
        "chosen": "富士山の標高は3776メートルで、日本最高峰の山です。",
        "rejected": "富士山はとても高い山です。具体的な数字はわかりません。",
    },
    {
        "prompt": "光合成とは何ですか？",
        "chosen": "光合成は植物が光エネルギーを使って二酸化炭素と水から有機物（糖）を作り出す過程です。",
        "rejected": "光合成は植物に関係することで、なんか難しいことをしています。",
    },
    {
        "prompt": "地球の年齢はどのくらいですか？",
        "chosen": "地球の年齢は約46億年と推定されています。放射性同位体の分析によって求められました。",
        "rejected": "地球はかなり古いと思います。何十億年かな？",
    },
    {
        "prompt": "俳句の音数を教えてください。",
        "chosen": "俳句は五・七・五の合計17音から構成されます。日本の伝統的な詩形式です。",
        "rejected": "俳句は短い詩です。音数はあまり重要ではないと思います。",
    },
    {
        "prompt": "新幹線の最高速度はどのくらいですか？",
        "chosen": "東海道新幹線の最高速度は時速285kmです。リニア中央新幹線は将来500km以上を予定しています。",
        "rejected": "新幹線は速い乗り物です。とても速く走ります。",
    },
    {
        "prompt": "ビタミンCを多く含む食品を教えてください。",
        "chosen": "ビタミンCを多く含む食品にはレモン、ピーマン、キウイフルーツ、ブロッコリーなどがあります。",
        "rejected": "果物や野菜にビタミンCが含まれています。特定の食品は思い出せません。",
    },
    {
        "prompt": "東日本大震災はいつ起きましたか？",
        "chosen": "東日本大震災は2011年3月11日に発生しました。マグニチュード9.0の巨大地震で、津波による甚大な被害をもたらしました。",
        "rejected": "東日本大震災は大きな地震でした。確か2000年代だったと思います。",
    },
    {
        "prompt": "素数とは何ですか？例を挙げてください。",
        "chosen": "素数とは1と自身以外に約数を持たない自然数です。例えば2、3、5、7、11、13などが素数です。",
        "rejected": "素数は特別な数のことです。2、4、6などが含まれます。",
    },
    {
        "prompt": "DNAの二重らせん構造を発見したのは誰ですか？",
        "chosen": "DNAの二重らせん構造は1953年にジェームズ・ワトソンとフランシス・クリックによって発見されました。",
        "rejected": "DNAの構造を発見した科学者がいたと思いますが、名前は思い出せません。",
    },
    {
        "prompt": "日本語の敬語の種類を教えてください。",
        "chosen": "日本語の敬語には、尊敬語（相手の行為を高める）、謙譲語（自分の行為を下げる）、丁寧語（丁寧に話す）の3種類があります。",
        "rejected": "日本語には丁寧な話し方があります。色々な種類がありますよね。",
    },
    {
        "prompt": "地球から月までの距離はどのくらいですか？",
        "chosen": "地球から月までの平均距離は約38万4400キロメートルです。光の速さで約1.3秒かかります。",
        "rejected": "月はとても遠いところにあります。何万キロかだと思います。",
    },
    {
        "prompt": "日本の人口はおよそ何人ですか？",
        "chosen": "日本の人口は2024年時点でおよそ1億2400万人です。少子高齢化により減少傾向にあります。",
        "rejected": "日本はたくさんの人が住んでいます。正確な数は知りません。",
    },
    {
        "prompt": "光の速さを教えてください。",
        "chosen": "光の速さは真空中で毎秒約29万9792キロメートル（約30万km/s）です。物理学の基本定数のひとつです。",
        "rejected": "光はとても速いです。すごい速さで移動します。",
    },
    {
        "prompt": "江戸時代はいつからいつまでですか？",
        "chosen": "江戸時代は1603年に徳川家康が江戸幕府を開いてから1868年の明治維新まで、約265年間続きました。",
        "rejected": "江戸時代は昔の日本の時代です。何百年か前のことだと思います。",
    },
    {
        "prompt": "光合成の化学式を教えてください。",
        "chosen": "光合成の化学式は 6CO₂ + 6H₂O + 光エネルギー → C₆H₁₂O₆ + 6O₂ です。二酸化炭素と水からグルコースと酸素が生成されます。",
        "rejected": "光合成の式は複雑で覚えていません。植物が光を使って何かを作ることは知っています。",
    },
    {
        "prompt": "フィボナッチ数列の最初の10項を教えてください。",
        "chosen": "フィボナッチ数列の最初の10項は 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 です。各項は前の2項の和です。",
        "rejected": "フィボナッチ数列は数字が並んでいます。1から始まって増えていきます。",
    },
    {
        "prompt": "日本国憲法はいつ施行されましたか？",
        "chosen": "日本国憲法は1947年5月3日に施行されました。この日は現在も「憲法記念日」として祝日になっています。",
        "rejected": "日本国憲法は戦後に作られました。詳しい日付は知りません。",
    },
    {
        "prompt": "人体の最大の臓器は何ですか？",
        "chosen": "人体の最大の臓器は皮膚です。成人の皮膚の面積は約1.5〜2平方メートルで、体を外部環境から守る役割を果たします。",
        "rejected": "人体の最大の臓器は心臓か肝臓だと思います。",
    },
    {
        "prompt": "インターネットはいつ発明されましたか？",
        "chosen": "インターネットの前身であるARPANETは1969年に開始されました。現在のWWW（World Wide Web）はティム・バーナーズ＝リーによって1989年に提案されました。",
        "rejected": "インターネットは最近発明されたと思います。詳しくは知りません。",
    },
    {
        "prompt": "地球温暖化の主な原因は何ですか？",
        "chosen": "地球温暖化の主な原因は、化石燃料の燃焼による二酸化炭素などの温室効果ガスの大気中濃度の増加です。産業革命以降に顕著になりました。",
        "rejected": "地球温暖化は環境問題です。人間の活動が原因だと聞いています。",
    },
    {
        "prompt": "ピタゴラスの定理を説明してください。",
        "chosen": "ピタゴラスの定理とは、直角三角形において、直角を挟む2辺の長さをa、b、斜辺をcとすると a² + b² = c² が成り立つという定理です。",
        "rejected": "ピタゴラスの定理は三角形に関する数学の定理です。詳しくは覚えていません。",
    },
    {
        "prompt": "ブラックホールとは何ですか？",
        "chosen": "ブラックホールは重力が非常に強く、光さえも脱出できない宇宙の天体です。大質量の星が超新星爆発を起こした後に形成されることがあります。",
        "rejected": "ブラックホールは宇宙にある穴のようなものです。とても危険です。",
    },
    {
        "prompt": "日本の国技は何ですか？",
        "chosen": "日本の国技は相撲です。2000年以上の歴史を持ち、年間6回の本場所が行われています。力士が土俵で技を競い合います。",
        "rejected": "日本の国技は柔道か剣道だと思います。確かではありませんが。",
    },
    {
        "prompt": "蒸気機関を発明したのは誰ですか？",
        "chosen": "実用的な蒸気機関を発明・改良したのはジェームズ・ワットです。1769年に効率的な蒸気機関の特許を取得し、産業革命を推進しました。",
        "rejected": "蒸気機関は昔の人が発明しました。名前は思い出せません。",
    },
    {
        "prompt": "正の整数nの階乗n!を定義してください。",
        "chosen": "n!（n階乗）とは1からnまでの全ての正の整数の積です。n! = n × (n-1) × ... × 2 × 1 と定義されます。例えば5! = 120です。",
        "rejected": "階乗は数学の概念です。記号は!を使います。",
    },
    {
        "prompt": "日本の三大急流を教えてください。",
        "chosen": "日本の三大急流は、富士川（静岡・山梨）、最上川（山形）、球磨川（熊本）です。",
        "rejected": "日本には急流の川がいくつかあります。具体的な名前は知りません。",
    },
    {
        "prompt": "人間の正常な体温はどのくらいですか？",
        "chosen": "人間の正常な体温は36.0〜37.0度程度です。37.5度以上を発熱と見なすことが多いです。",
        "rejected": "人間の体温は普通の温度です。熱があるかどうかで判断します。",
    },
    {
        "prompt": "スマートフォンと従来の携帯電話の主な違いを教えてください。",
        "chosen": "スマートフォンはタッチスクリーン、インターネット接続、アプリのインストールが可能で、小型コンピュータとして機能します。従来の携帯電話は通話・SMS中心で機能が限定的でした。",
        "rejected": "スマートフォンは最新の電話で、古い携帯電話より便利です。",
    },
    {
        "prompt": "三権分立とは何ですか？",
        "chosen": "三権分立とは立法権（国会）、行政権（内閣）、司法権（裁判所）を分離して相互に監視・抑制することで、権力の集中を防ぐ政治原理です。",
        "rejected": "三権分立は政治の仕組みです。権力を分けることが大切です。",
    },
]


def format_pair(prompt, response, tokenizer, max_length=256):
    text = f"{prompt}{response}{tokenizer.eos_token}"
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.samples = []
        for item in data:
            chosen_ids, chosen_mask = format_pair(item["prompt"], item["chosen"], tokenizer, max_length)
            rejected_ids, rejected_mask = format_pair(item["prompt"], item["rejected"], tokenizer, max_length)
            self.samples.append({
                "chosen_input_ids": chosen_ids,
                "chosen_attention_mask": chosen_mask,
                "rejected_input_ids": rejected_ids,
                "rejected_attention_mask": rejected_mask,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_log_probs(model, input_ids, attention_mask):
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].float()
    return (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=0.1):
    chosen_ratio = policy_chosen_logp - ref_chosen_logp
    rejected_ratio = policy_rejected_logp - ref_rejected_logp
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()
    reward_margin = (chosen_ratio - rejected_ratio).mean().item()
    return loss, reward_margin


class DPOTrainer:
    def __init__(self, policy_model, ref_model, tokenizer, dataset, device, beta=0.1, lr=1e-5, batch_size=4):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.beta = beta
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_epoch(self, epoch):
        self.policy.train()
        self.ref.eval()
        total_loss = 0.0
        total_margin = 0.0
        steps = 0

        for batch in self.loader:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_mask = batch["chosen_attention_mask"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_mask = batch["rejected_attention_mask"].to(self.device)

            policy_chosen_logp = get_log_probs(self.policy, chosen_ids, chosen_mask)
            policy_rejected_logp = get_log_probs(self.policy, rejected_ids, rejected_mask)

            with torch.no_grad():
                ref_chosen_logp = get_log_probs(self.ref, chosen_ids, chosen_mask)
                ref_rejected_logp = get_log_probs(self.ref, rejected_ids, rejected_mask)

            loss, margin = dpo_loss(
                policy_chosen_logp, policy_rejected_logp,
                ref_chosen_logp, ref_rejected_logp,
                beta=self.beta,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_margin += margin
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        avg_margin = total_margin / max(steps, 1)
        print(f"Epoch {epoch+1} | DPO Loss: {avg_loss:.4f} | Reward Margin: {avg_margin:.4f}")
        return avg_loss, avg_margin


def main():
    model_name = "rinna/japanese-gpt2-medium"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    for param in ref_model.parameters():
        param.requires_grad = False

    dataset = PreferenceDataset(PREFERENCE_DATA, tokenizer)

    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        beta=0.1,
        lr=1e-5,
        batch_size=4,
    )

    print("\n=== DPO Training ===")
    losses = []
    margins = []
    for epoch in range(2):
        loss, margin = trainer.train_epoch(epoch)
        losses.append(loss)
        margins.append(margin)

    print("\n=== Training Summary ===")
    print(f"Loss:   {losses[0]:.4f} → {losses[-1]:.4f} ({'decreased' if losses[-1] < losses[0] else 'increased'})")
    print(f"Margin: {margins[0]:.4f} → {margins[-1]:.4f} ({'improved' if margins[-1] > margins[0] else 'decreased'})")

    out_dir = os.path.dirname(__file__)
    policy_model.save_pretrained(os.path.join(out_dir, "dpo_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "dpo_model"))
    print(f"\nModel saved to {out_dir}/dpo_model")


if __name__ == "__main__":
    main()
