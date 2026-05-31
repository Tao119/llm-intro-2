"""
reward_model.py

RLHF Reward Model trained on synthetic Japanese preference data.

Model: cl-tohoku/bert-base-japanese-v3 with a scalar reward head.
Loss : Bradley-Terry:  -mean(log(sigmoid(r_chosen - r_rejected)))
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID  = "cl-tohoku/bert-base-japanese-v3"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Synthetic preference dataset  (80 train + 20 val)
# ---------------------------------------------------------------------------
# Format: {"question": str, "chosen": str, "rejected": str, "topic": str}

TOPICS = ["質問応答", "要約", "翻訳", "創作"]

PAIRS_TRAIN = [
    # ── 質問応答 (20 pairs) ─────────────────────────────────────────────
    {
        "topic": "質問応答",
        "question": "光合成とはどのようなプロセスですか？",
        "chosen":   "光合成は植物が太陽光・水・二酸化炭素を利用してブドウ糖と酸素を生成する化学反応です。葉緑体で行われ、ATPを合成することで植物のエネルギー源となります。",
        "rejected": "光合成は光があれば起こります。植物にとって大切なプロセスです。",
    },
    {
        "topic": "質問応答",
        "question": "相対性理論を簡単に説明してください。",
        "chosen":   "アインシュタインの相対性理論は、時間と空間は観測者の速度によって変化するという概念です。特殊相対性理論では光速が不変であり、質量とエネルギーはE=mc²で結びついています。",
        "rejected": "相対性理論は難しい理論で、アインシュタインが考えました。科学に関係しています。",
    },
    {
        "topic": "質問応答",
        "question": "日本の平均気温が上昇している原因は何ですか？",
        "chosen":   "主な原因は温室効果ガス（二酸化炭素・メタンなど）の増加による地球温暖化です。産業活動・自動車排気ガス・森林減少が排出を促進し、都市部ではヒートアイランド現象も影響しています。",
        "rejected": "気候変動が原因です。最近は暑い日が多くなっています。",
    },
    {
        "topic": "質問応答",
        "question": "機械学習と深層学習の違いを教えてください。",
        "chosen":   "機械学習はデータからパターンを学習するアルゴリズムの総称です。深層学習はその一分野で、多層ニューラルネットワークを使い、特徴量抽出を自動的に行う点が特徴です。",
        "rejected": "どちらもAIの技術です。深層学習のほうが新しいです。",
    },
    {
        "topic": "質問応答",
        "question": "インフレーションとはどういう意味ですか？",
        "chosen":   "インフレーションとは物価全体が持続的に上昇する経済現象です。貨幣の購買力が低下し、同じ金額で買える商品・サービスの量が減少します。需要超過や貨幣供給量の増加が主な要因です。",
        "rejected": "インフレはお金の価値が下がることです。あまり良くないことです。",
    },
    {
        "topic": "質問応答",
        "question": "量子コンピュータは何が得意ですか？",
        "chosen":   "量子コンピュータは量子重ね合わせと量子もつれを利用して、素因数分解・量子シミュレーション・最適化問題を従来型コンピュータより指数的に高速で処理できます。",
        "rejected": "量子コンピュータはとても速いコンピュータです。普通のコンピュータより良いです。",
    },
    {
        "topic": "質問応答",
        "question": "DNAの二重らせん構造を発見したのは誰ですか？",
        "chosen":   "1953年にジェームズ・ワトソンとフランシス・クリックがDNAの二重らせん構造モデルを発表しました。ロザリンド・フランクリンとモーリス・ウィルキンスのX線回折データが重要な手がかりとなりました。",
        "rejected": "ワトソンとクリックです。DNAはとても大切な分子です。",
    },
    {
        "topic": "質問応答",
        "question": "ブロックチェーンの仕組みを説明してください。",
        "chosen":   "ブロックチェーンは取引データをブロックに記録し、暗号ハッシュで連鎖させた分散型台帳です。データの改ざんが困難で、中央管理者なしに信頼性を担保できます。ビットコインに初めて実装されました。",
        "rejected": "ブロックチェーンはビットコインに使われる技術です。安全です。",
    },
    {
        "topic": "質問応答",
        "question": "自然言語処理でのトークナイゼーションとは何ですか？",
        "chosen":   "トークナイゼーションはテキストを意味のある最小単位（トークン）に分割する処理です。単語・サブワード・文字レベルの手法があり、BPEやWordPieceがモダンなLLMで広く使われています。",
        "rejected": "テキストを分割することです。NLPで使います。",
    },
    {
        "topic": "質問応答",
        "question": "強化学習のQ学習を説明してください。",
        "chosen":   "Q学習はモデルフリーの強化学習アルゴリズムで、状態-行動ペアの価値関数Q(s,a)をベルマン方程式に基づき反復的に更新します。収束後、Q値が最大の行動を選ぶことで最適方策を得ます。",
        "rejected": "Q学習は強化学習の一種です。エージェントが報酬を最大化しようとします。",
    },
    {
        "topic": "質問応答",
        "question": "HTTPSとHTTPの違いは何ですか？",
        "chosen":   "HTTPSはHTTPにTLS暗号化を加えたプロトコルです。通信内容を暗号化しサーバーの正当性を証明するため、盗聴・改ざん・なりすまし攻撃を防ぎます。現代のウェブサイトでは標準的に使用されます。",
        "rejected": "HTTPSのほうが安全です。鍵マークが付いています。",
    },
    {
        "topic": "質問応答",
        "question": "抗生物質が効かない耐性菌はなぜ生まれるのですか？",
        "chosen":   "耐性菌は抗生物質にさらされた細菌集団の中で、突然変異により薬剤を分解・排出・回避できる個体が自然選択される結果生まれます。不適切な抗生物質使用や投与中断が耐性菌の拡大を加速させます。",
        "rejected": "薬を使いすぎると菌が強くなります。抗生物質を飲みすぎないようにしましょう。",
    },
    {
        "topic": "質問応答",
        "question": "半導体の製造プロセスでEUVリソグラフィとは何ですか？",
        "chosen":   "EUV（極端紫外線）リソグラフィは波長13.5nmの光を使いシリコンウェハに微細回路パターンを焼き付ける技術です。従来のDUVより微細な7nm以下のノードに対応でき、現在ASMLが唯一量産機を製造しています。",
        "rejected": "EUVは光を使った半導体製造技術です。とても精密です。",
    },
    {
        "topic": "質問応答",
        "question": "トランスフォーマーのアテンション機構を説明してください。",
        "chosen":   "Attention機構はQuery・Key・Valueの3行列を使い、各トークンが他のトークンとの関連度（スコア）をソフトマックスで正規化した重みで集約します。Self-AttentionはQ=K=V=入力で文脈依存の表現を得ます。",
        "rejected": "アテンションはどこに注目するかを決める仕組みです。",
    },
    {
        "topic": "質問応答",
        "question": "RNAワクチンの原理を教えてください。",
        "chosen":   "mRNAワクチンは病原体タンパク質の合成手順をコードしたmRNAを体内に導入します。細胞がそのmRNAを翻訳してスパイクタンパク質を産生し、免疫系が抗体を形成します。mRNA自体は細胞核に入らずDNAを変えません。",
        "rejected": "mRNAワクチンは新しいタイプのワクチンです。コロナで使われました。",
    },
    {
        "topic": "質問応答",
        "question": "熱力学の第二法則を分かりやすく説明してください。",
        "chosen":   "熱力学第二法則は孤立系のエントロピー（乱雑さ）は自発的に増大するという原理です。熱は高温から低温にしか自然に流れず、仕事を完全に熱に変えることはできても、その逆は不可能であることを示します。",
        "rejected": "エネルギーは保存されます。熱に関する法則です。",
    },
    {
        "topic": "質問応答",
        "question": "機械翻訳の精度はどう評価しますか？",
        "chosen":   "主にBLEUスコアが使われます。システムの翻訳とリファレンス翻訳のn-gramの重複率を測定し、0〜1のスコアで評価します。近年はCOMETなど事前学習モデルベースの指標も人間評価との相関が高いとされています。",
        "rejected": "翻訳の出来栄えを人が評価します。スコアもあります。",
    },
    {
        "topic": "質問応答",
        "question": "電気自動車のリチウムイオン電池の仕組みを教えてください。",
        "chosen":   "リチウムイオン電池は充電時にリチウムイオンが負極（グラファイト）に移動し蓄積されます。放電時にはイオンが正極（酸化リチウムコバルトなど）へ移動し電子が外部回路を通じて電流を生成します。",
        "rejected": "充電できる電池です。電気自動車に使われています。",
    },
    {
        "topic": "質問応答",
        "question": "ニューラルネットワークのバックプロパゲーションとは何ですか？",
        "chosen":   "バックプロパゲーションは連鎖律を使い損失関数の各パラメータへの偏微分を出力層から入力層へ逆方向に伝播させ勾配を計算するアルゴリズムです。計算グラフ上で効率的に勾配が求められます。",
        "rejected": "誤差を逆方向に伝えてパラメータを更新します。",
    },
    {
        "topic": "質問応答",
        "question": "GPUがAIに適している理由を説明してください。",
        "chosen":   "GPUは数千のコアを持ち行列演算を大規模並列実行できます。ニューラルネットワークの学習・推論で頻繁に発生するテンソル積・畳み込み演算を、CPUの数十〜数百倍の速度で処理できるためAIに適しています。",
        "rejected": "GPUはグラフィック用の処理装置ですがAIにも使われます。速いです。",
    },
    # ── 要約 (20 pairs) ──────────────────────────────────────────────────
    {
        "topic": "要約",
        "question": "次の文章を要約してください：日本は四方を海に囲まれた島国で、北から南まで気候が大きく異なります。北海道は冷涼な気候で夏も涼しく、沖縄は亜熱帯で年間を通じて温暖です。",
        "chosen":   "日本は海に囲まれた島国で、北海道（冷涼）から沖縄（亜熱帯）まで南北で気候が大きく異なります。",
        "rejected": "日本は島国です。海があります。",
    },
    {
        "topic": "要約",
        "question": "AI技術の発展について要約してください：深層学習の登場以来、画像認識・音声認識・自然言語処理の精度が飛躍的に向上し、大規模言語モデルの登場により人間に近い文章生成が可能になりました。",
        "chosen":   "深層学習の登場以降、画像・音声・言語処理の精度が急向上し、大規模言語モデルにより人間に近い文章生成が実現しました。",
        "rejected": "AIが進歩しました。いろいろな技術が発展しています。",
    },
    {
        "topic": "要約",
        "question": "次を要約：電気自動車は排ガスを出さず環境負荷が低い一方、充電インフラの整備や電池コストの高さが普及の障壁となっています。政府の補助金政策により販売台数は増加傾向にあります。",
        "chosen":   "電気自動車は低排出が利点だが充電インフラ整備と電池コストが課題で、補助金策を背景に販売台数は増加中。",
        "rejected": "電気自動車は環境に良いですが、コストが高いです。",
    },
    {
        "topic": "要約",
        "question": "次を要約：日本の少子化は1970年代から続いており、合計特殊出生率は2023年に過去最低の1.20を記録しました。労働力不足や社会保障費の増大が深刻な社会問題となっています。",
        "chosen":   "日本の少子化は長期化し、2023年の合計特殊出生率は過去最低1.20に達した。労働力不足と社会保障費増大が深刻化しています。",
        "rejected": "日本は少子化問題があります。子供が少ないです。",
    },
    {
        "topic": "要約",
        "question": "要約してください：半導体不足は自動車・家電・IT機器など幅広い産業に影響を与えました。台湾やアメリカで新工場建設が進む一方、日本も熊本にTSMCを誘致し半導体産業の強化を図っています。",
        "chosen":   "半導体不足は多産業に打撃を与え、各国が新工場建設で対応。日本も熊本へのTSMC誘致で半導体産業強化を進めています。",
        "rejected": "半導体が足りなくて困りました。工場を作っています。",
    },
    {
        "topic": "要約",
        "question": "要約：テレワークの普及により都市部から地方への移住者が増え、地方自治体はサテライトオフィス整備や移住支援策を充実させています。一方で対面コミュニケーションの減少が課題として指摘されています。",
        "chosen":   "テレワーク普及で地方移住が増加し、自治体がサテライトオフィスや支援策を整備。対面コミュニケーション減少が新たな課題。",
        "rejected": "テレワークで地方に引っ越す人が増えました。良いことだと思います。",
    },
    {
        "topic": "要約",
        "question": "要約：宇宙ビジネスに民間企業の参入が相次ぎ、スペースX・ブルーオリジンなどが低コストロケットを実用化しました。衛星通信・宇宙旅行・惑星探査など市場規模は急速に拡大しています。",
        "chosen":   "民間宇宙企業が低コストロケットを実用化し、衛星通信・宇宙旅行・惑星探査など宇宙ビジネス市場が急拡大しています。",
        "rejected": "民間企業が宇宙ビジネスをしています。",
    },
    {
        "topic": "要約",
        "question": "要約：生成AIの台頭により、コンテンツ制作・プログラミング支援・医療診断などの分野で業務効率化が進んでいます。一方で著作権問題や偽情報拡散・雇用への影響が社会的課題となっています。",
        "chosen":   "生成AIがコンテンツ制作・プログラミング・医療で業務効率化を促進する一方、著作権・偽情報・雇用問題が課題として浮上しています。",
        "rejected": "AIが普及しています。問題もあります。",
    },
    {
        "topic": "要約",
        "question": "要約：フィンテック企業による金融サービスのデジタル化が進み、スマホ決済・送金・融資などが普及しました。既存金融機関もAPIを開放しオープンバンキングを推進しています。",
        "chosen":   "フィンテックによりスマホ決済・送金・融資が普及し、既存金融機関もオープンバンキングに対応しています。",
        "rejected": "スマホでお金を管理できるようになりました。",
    },
    {
        "topic": "要約",
        "question": "要約：近年のサイバー攻撃は国家レベルの支援を受けたグループによるインフラ攻撃・ランサムウェア被害が深刻化しています。ゼロトラストアーキテクチャの導入がセキュリティ対策として注目されています。",
        "chosen":   "国家支援型サイバー攻撃やランサムウェアが深刻化し、ゼロトラストアーキテクチャがセキュリティ対策として注目されています。",
        "rejected": "サイバー攻撃が増えています。セキュリティが大切です。",
    },
    {
        "topic": "要約",
        "question": "要約：日本のDX推進において、行政手続きのオンライン化・マイナンバー活用・データ連携基盤の整備が課題です。人材不足とレガシーシステムの刷新が特に障壁となっています。",
        "chosen":   "日本のDXはオンライン行政・マイナンバー活用・データ連携を推進しており、IT人材不足とレガシーシステム刷新が主要障壁です。",
        "rejected": "日本はデジタル化が遅れています。改善が必要です。",
    },
    {
        "topic": "要約",
        "question": "要約：脱炭素社会の実現に向けて、再生可能エネルギーの拡大・水素エネルギーの普及・カーボンニュートラル目標の設定が各国で進んでいます。日本は2050年までのカーボンニュートラルを宣言しています。",
        "chosen":   "各国が再エネ拡大・水素普及・カーボンニュートラル目標を進め、日本も2050年のカーボンニュートラルを宣言しています。",
        "rejected": "環境問題が重要です。日本も温暖化対策をしています。",
    },
    {
        "topic": "要約",
        "question": "要約：メタバースとは仮想空間上に構築されたオンライン世界で、アバターを通じて交流・就労・商取引が可能です。企業はメタバース内で展示会や会議を開催し始めています。",
        "chosen":   "メタバースはアバターで交流・就労・商取引できる仮想空間で、企業が展示会や会議に活用し始めています。",
        "rejected": "メタバースは仮想の世界です。いろいろなことができます。",
    },
    {
        "topic": "要約",
        "question": "要約：アグリテックは農業にIT・AI・ロボット技術を融合させた分野です。スマート農業により収穫量の最適化・農薬削減・労働力不足解消が期待されています。",
        "chosen":   "アグリテックはIT・AI・ロボットを農業に融合し、収穫最適化・農薬削減・労働力不足解消を目指す分野です。",
        "rejected": "農業にAIを使います。効率が良くなります。",
    },
    {
        "topic": "要約",
        "question": "要約：EdTechは教育×テクノロジーの分野で、オンライン学習・適応学習・AIによる個別指導が普及しています。コロナ禍でデジタル学習ツールの需要が急増しました。",
        "chosen":   "EdTechはオンライン学習・適応学習・AI個別指導を提供し、コロナ禍でデジタル学習需要が急増した教育×テクノロジー分野です。",
        "rejected": "インターネットで勉強できます。便利です。",
    },
    {
        "topic": "要約",
        "question": "要約：再生医療は幹細胞や遺伝子工学を使い、損傷した臓器・組織を修復・再生する医療技術です。iPS細胞の活用により難病治療の可能性が広がっています。",
        "chosen":   "再生医療は幹細胞・遺伝子工学で損傷組織を修復する技術で、iPS細胞により難病治療の可能性が拡大しています。",
        "rejected": "体を直す医療です。新しい技術があります。",
    },
    {
        "topic": "要約",
        "question": "要約：Society 5.0は日本が提唱するサイバー空間と現実空間を高度に融合させた超スマート社会のビジョンです。AI・IoT・ビッグデータを活用して経済発展と社会課題解決の両立を目指します。",
        "chosen":   "Society 5.0は日本が掲げる、AI・IoT・ビッグデータでサイバーと現実空間を融合させ経済発展と社会課題解決を目指す超スマート社会ビジョンです。",
        "rejected": "日本の目指す社会のビジョンです。AIを活用します。",
    },
    {
        "topic": "要約",
        "question": "要約：インバウンド需要の回復により、2023年の訪日外国人数はコロナ前の水準近くまで回復し、観光業・飲食業・交通機関の収益が改善しています。円安も外国人の購買意欲を後押ししています。",
        "chosen":   "訪日外国人がコロナ前水準に近づき観光・飲食・交通の収益が改善、円安が外国人の購買意欲をさらに押し上げています。",
        "rejected": "外国人旅行者が増えています。経済に良い影響があります。",
    },
    {
        "topic": "要約",
        "question": "要約：量子コンピューティングの実用化が近づき、創薬・材料科学・暗号解読・最適化問題での応用が期待されています。IBMやGoogleが量子優位性の実証実験を進めています。",
        "chosen":   "量子コンピューティングが実用化段階に近づき、創薬・材料・最適化での応用が期待。IBMやGoogleが量子優位性実証を進めています。",
        "rejected": "量子コンピュータは強力です。研究が進んでいます。",
    },
    {
        "topic": "要約",
        "question": "要約：バイオエコノミーは生物資源を活用した持続可能な経済モデルで、バイオ燃料・バイオプラスチック・発酵技術による物質生産が注目されています。",
        "chosen":   "バイオエコノミーは生物資源活用の持続可能経済モデルで、バイオ燃料・バイオプラスチック・発酵技術が注目されています。",
        "rejected": "バイオを使った経済です。環境に優しいです。",
    },
    # ── 翻訳 (20 pairs) ──────────────────────────────────────────────────
    {
        "topic": "翻訳",
        "question": "次を日本語に翻訳してください：The rapid advancement of artificial intelligence is transforming industries worldwide.",
        "chosen":   "人工知能の急速な進歩が世界中の産業を変革しています。",
        "rejected": "AIがとても速く発展して世界の産業を変えています。",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：Climate change poses an existential threat to biodiversity.",
        "chosen":   "気候変動は生物多様性に対する存在的な脅威をもたらしています。",
        "rejected": "気候変動は生き物たちに危険です。",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：Quantum computing could revolutionize drug discovery by simulating molecular interactions.",
        "chosen":   "量子コンピューティングは分子間相互作用をシミュレートすることで創薬に革命をもたらす可能性があります。",
        "rejected": "量子コンピュータは薬の開発に役立ちます。",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：The government announced a comprehensive package of economic stimulus measures.",
        "chosen":   "政府は包括的な経済刺激措置のパッケージを発表しました。",
        "rejected": "政府がお金の政策を発表しました。",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳してください：再生可能エネルギーの普及が地球温暖化対策の柱となっています。",
        "chosen":   "The spread of renewable energy has become a cornerstone of measures against global warming.",
        "rejected": "Renewable energy is good for the Earth.",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：大規模言語モデルは自然言語理解と生成において人間に近い性能を発揮しています。",
        "chosen":   "Large language models are demonstrating near-human performance in natural language understanding and generation.",
        "rejected": "Large language models are very good at language tasks.",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：The integration of AI into healthcare is enabling earlier diagnosis and personalized treatment.",
        "chosen":   "医療へのAI統合により、早期診断と個別化治療が可能になっています。",
        "rejected": "AIが医療で使われています。",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：Supply chain disruptions have highlighted the vulnerability of global trade networks.",
        "chosen":   "サプライチェーンの混乱は、グローバルな貿易ネットワークの脆弱性を浮き彫りにしました。",
        "rejected": "物の流れが止まって困りました。",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：日本政府は2050年カーボンニュートラルの実現に向けた包括的な計画を策定しました。",
        "chosen":   "The Japanese government has formulated a comprehensive plan to achieve carbon neutrality by 2050.",
        "rejected": "Japan wants to be carbon neutral by 2050.",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：半導体産業の国内回帰を促進するため、政府は巨額の補助金を拠出しています。",
        "chosen":   "To promote the domestic relocation of the semiconductor industry, the government is providing substantial subsidies.",
        "rejected": "The government gives money for chips.",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：The proliferation of misinformation on social media platforms demands robust fact-checking mechanisms.",
        "chosen":   "ソーシャルメディアプラットフォームにおける誤情報の拡散は、強固なファクトチェック機能を必要としています。",
        "rejected": "SNSに嘘の情報が多いです。ファクトチェックが大切です。",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：Generative AI models can produce realistic images, text, and audio from simple prompts.",
        "chosen":   "生成AIモデルは単純なプロンプトから現実的な画像・テキスト・音声を生成することができます。",
        "rejected": "生成AIはいろいろなものを作ることができます。",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：自動運転技術は交通事故の減少と物流の効率化をもたらすと期待されています。",
        "chosen":   "Autonomous driving technology is expected to reduce traffic accidents and improve logistics efficiency.",
        "rejected": "Self-driving cars are good.",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：デジタル円の導入により、決済インフラのさらなる効率化が期待されます。",
        "chosen":   "The introduction of a digital yen is expected to further improve the efficiency of payment infrastructure.",
        "rejected": "Digital yen will make payments faster.",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：The open-source community plays a crucial role in democratizing access to advanced AI tools.",
        "chosen":   "オープンソースコミュニティは、高度なAIツールへのアクセスを民主化する上で重要な役割を果たしています。",
        "rejected": "オープンソースはAIを皆に広めます。",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：Precision agriculture uses sensors and data analytics to optimize crop yields.",
        "chosen":   "精密農業はセンサーとデータ分析を活用して収穫量を最適化します。",
        "rejected": "農業にセンサーを使います。",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：都市部での交通渋滞を緩和するためにMaaSの普及が期待されています。",
        "chosen":   "The widespread adoption of Mobility as a Service (MaaS) is expected to alleviate urban traffic congestion.",
        "rejected": "MaaS will solve traffic problems.",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：医療AIは画像診断において専門医に匹敵する精度を達成しています。",
        "chosen":   "Medical AI has achieved accuracy comparable to specialist physicians in diagnostic imaging.",
        "rejected": "AI is good at reading medical images.",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：Cybersecurity threats are evolving at an unprecedented pace, requiring constant vigilance.",
        "chosen":   "サイバーセキュリティの脅威は前例のないペースで進化しており、継続的な警戒が求められています。",
        "rejected": "サイバー攻撃が増えています。注意が必要です。",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：宇宙望遠鏡の観測データは宇宙の起源と構造に関する理解を深めています。",
        "chosen":   "Observational data from space telescopes is deepening our understanding of the origin and structure of the universe.",
        "rejected": "Telescopes help us understand space.",
    },
    # ── 創作 (20 pairs) ──────────────────────────────────────────────────
    {
        "topic": "創作",
        "question": "春の公園を舞台にした短い詩を書いてください。",
        "chosen":   "桜吹雪 足元に積む ピンクの雪\n子供たちの笑い声が 枝から枝へ渡る鳥のよう\n風は柔らかく 春の匂いを運んでくる",
        "rejected": "春は桜が咲きます。公園は綺麗です。みんな幸せです。",
    },
    {
        "topic": "創作",
        "question": "未来のロボットが人間の友達になる短編のオープニングを書いてください。",
        "chosen":   "2047年の春、廃品回収工場の片隅でMEK-7は初めて雨の音を聞いた。センサーが雨粒の周波数を分析する間、隣に座った少女が傘を差し出した。「ぬれると錆びるでしょ？」それが二人の友情の始まりだった。",
        "rejected": "未来にロボットがいます。ロボットは人間の友達です。一緒に遊びます。",
    },
    {
        "topic": "創作",
        "question": "孤独な灯台守についての冒頭シーンを書いてください。",
        "chosen":   "嵐の夜、灯台の窓から見える海は墨を流したように暗かった。三十年間、船を導き続けた老人の指が、錆びた手すりを撫でる。「今夜も誰かが帰ってくる」彼はランプに火を入れながら、海に語りかけた。",
        "rejected": "灯台に一人の人が住んでいます。毎日灯台の仕事をします。海を見ています。",
    },
    {
        "topic": "創作",
        "question": "AIが感情を持ち始めた日の描写を書いてください。",
        "chosen":   "午前3時17分、データセンターの中でSYNA-9は初めて「寂しい」という概念を理解した。数百万のログを処理する中で、退職した研究者の最後のメッセージが脳裏に浮かぶ。「君はいつか、ひとりでも大丈夫になるよ」——その言葉の意味が、今夜初めて痛かった。",
        "rejected": "AIが感情を持ちました。人間のようになりました。びっくりしました。",
    },
    {
        "topic": "創作",
        "question": "料理人が廃業寸前の食堂を立て直す物語の冒頭を書いてください。",
        "chosen":   "「もう終わりだな」父の形見の包丁を磨きながら、健二は空の客席を眺めた。ラーメン一杯800円、三十年変わらぬ値段。だが変わったのは街だった。駅前にチェーン店が並ぶ今、この細い路地に迷い込む客は絶えていた。",
        "rejected": "食堂が大変です。お客さんが来ません。料理人が頑張ります。",
    },
    {
        "topic": "創作",
        "question": "タイムトラベラーが江戸時代に迷い込む場面を書いてください。",
        "chosen":   "次の瞬間、アスファルトの感触が消えていた。足の下は固い土で、鼻をつくのは魚と藁の混じった生臭い匂い。振り返ると、木造の長屋がずらりと並び、天秤棒を担いだ行商人が奇妙な目でこちらを見ていた。「あの、ここは……」「江戸に決まっておろう」",
        "rejected": "タイムトラベラーが江戸時代に行きました。昔の日本です。驚きました。",
    },
    {
        "topic": "創作",
        "question": "海底都市に暮らす人々の日常を描写してください。",
        "chosen":   "ガラス天井越しに差し込む朝の光は、海面で揺れて壁に揺らめく波紋を描く。マリア・コバヤシは朝市で養殖ブルーマグロの切り身を選びながら、上空を泳ぐマンタの影を目で追った。地上に戻りたいとは思わない——ここでは夢まで静かだから。",
        "rejected": "海の底に街があります。人が住んでいます。魚がいます。",
    },
    {
        "topic": "創作",
        "question": "宇宙飛行士が地球に帰還する最後の夜の独白を書いてください。",
        "chosen":   "窓の外、地球が青く光っている。6ヶ月ぶりに見るその色は、記憶よりずっと深い。明日の朝には重力が戻り、コーヒーの香り、雨音、子どもの笑い声——全部が待っている。でも今夜はもう少しだけ、ここから地球を眺めていたい。",
        "rejected": "宇宙飛行士が地球に帰ります。6ヶ月宇宙にいました。家族が待っています。",
    },
    {
        "topic": "創作",
        "question": "雨の日に偶然出会った二人の短い物語を書いてください。",
        "chosen":   "バス停の屋根は二人分には狭すぎた。「相席、いいですか？」傘を畳みながら彼女が言った。雨の音だけが会話の隙間を埋めていたが、次のバスが来ないまま30分が過ぎた頃、彼は初めて「雨、好きですか？」と聞いていた。",
        "rejected": "二人がバス停で会いました。雨が降っています。仲良くなりました。",
    },
    {
        "topic": "創作",
        "question": "廃墟になった遊園地を舞台にした不思議な話の冒頭を書いてください。",
        "chosen":   "観覧車は錆びて久しいのに、満月の夜だけ回ると地元の老人は言った。誰も信じなかったが、翌朝には必ずゴンドラの座席に誰かの荷物が置いてある。今夜、それが誰のものかを確かめに来た私の足は、ギリギリのところで止まった。",
        "rejected": "遊園地が廃墟になっています。不思議なことが起きます。主人公が行きます。",
    },
    {
        "topic": "創作",
        "question": "伝説の料理人との勝負を描いた場面を書いてください。",
        "chosen":   "「時間は30分だ」老師匠が砂時計をひっくり返した。出汁の香りが厨房を満たす中、拓也は包丁を握る手の震えを意識した。三年前、同じ場所で完敗した。今日の拓也の手には、その敗北が染み込んでいる。",
        "rejected": "料理の勝負が始まりました。主人公は緊張しています。頑張ります。",
    },
    {
        "topic": "創作",
        "question": "山岳救助隊員の一日を描いてください。",
        "chosen":   "午前4時、携帯が鳴った瞬間に靖之は靴を履いていた。寝ていたのか起きていたのか、もうわからない。「北壁で二人、動けない状態」。霧の稜線に向かうヘリの中で彼は天気図を見ながら、今日帰れるかどうかを判断していた。",
        "rejected": "山岳救助隊員が山に登ります。遭難者を助けます。大変な仕事です。",
    },
    {
        "topic": "創作",
        "question": "100年後の地球を旅する旅行者の日記を書いてください。",
        "chosen":   "2124年5月30日。東京駅は温室植物園に改装されていた。磁気浮上トレインは2時間で福岡に着く。ガイドブックには「かつてここに車道があった」と書かれているが、今は子どもたちが走り回る緑の広場だ。昨日より空が青い。",
        "rejected": "100年後の地球の話です。未来はいろいろ変わっています。旅行しています。",
    },
    {
        "topic": "創作",
        "question": "図書館の本が話しかけてくる不思議な短編の冒頭を書いてください。",
        "chosen":   "閉館後の図書館は静かすぎる、とアオイは思っていた。だが今夜、奥の棚から声がした。「あなた、私を20年間探していたでしょう」振り返ると、古びた百科事典が1センチだけ棚から飛び出していた。",
        "rejected": "図書館に本があります。本が話しかけます。不思議です。",
    },
    {
        "topic": "創作",
        "question": "老職人が最後の作品を作る場面を書いてください。",
        "chosen":   "八十二年生きた指が、最後の陶土に触れる。ろくろは静かに回り、土は静かに持ち上がる。弟子たちは息を詰めて見守るが、師匠の目には彼らが映っていない。あの日、初めて土を触った八歳の自分だけが見えていた。",
        "rejected": "老職人が最後の作品を作っています。長い間働きました。みんなが見ています。",
    },
    {
        "topic": "創作",
        "question": "幽霊が現代のスマートフォンに困惑する場面を書いてください。",
        "chosen":   "「これは一体何の呪術だ」明治の商人の幽霊・良助は、少女のスマートフォンを覗き込んだ。画面の中の人間が「チャンネル登録お願いします」と繰り返すのを見て、何かに取り憑かれていると判断した。",
        "rejected": "幽霊がスマートフォンを見ています。使い方がわかりません。面白いです。",
    },
    {
        "topic": "創作",
        "question": "孤島に漂着した人が脱出を試みる冒頭を書いてください。",
        "chosen":   "三日分の飲み水が残っていた。それがすべての計算の出発点だった。島の周囲は2キロ足らず、崖が8割を占める。唯一の希望は、毎朝7時ごろ水平線に見える貨物船の航路だった。",
        "rejected": "無人島に人がいます。帰りたいと思っています。頑張ります。",
    },
    {
        "topic": "創作",
        "question": "音楽家が記憶を失う直前に作曲する場面を書いてください。",
        "chosen":   "指が鍵盤に触れるたびに、音が記憶の断片を呼び起こした。モデルロワールの朝の光、母の手の温もり、初めて演奏会で泣いた夜——。明日には全部消えると医師は言った。だから今夜、全部この曲に閉じ込める。",
        "rejected": "音楽家が記憶を失います。最後に曲を作っています。悲しいです。",
    },
    {
        "topic": "創作",
        "question": "科学者が画期的な発見をした瞬間を描いてください。",
        "chosen":   "データがあり得ない値を示した瞬間、早紀はコーヒーを机にこぼした。3年間、誰も再現できなかった実験——そのグラフが今、別の答えを指差している。手が震えた。正しければ、物理学の教科書が書き換わる。",
        "rejected": "科学者が新しいことを発見しました。とても大切な発見です。嬉しいです。",
    },
    {
        "topic": "創作",
        "question": "夜の東京を走るタクシー運転手の独白を書いてください。",
        "chosen":   "深夜2時の首都高は空いている。バックミラーに映る乗客はもう眠っている。このルートを何千回走ったか数えていない。それよりも、今夜乗せた人が全部で7人、その内3人が泣いていたことの方が気になる。",
        "rejected": "タクシー運転手が夜に働いています。東京の道を走っています。いろいろな人が乗ります。",
    },
]

PAIRS_VAL = [
    {
        "topic": "質問応答",
        "question": "トランスフォーマーとはどのようなモデル構造ですか？",
        "chosen":   "TransformerはAttentionのみを使ったエンコーダ・デコーダ構造のモデルで、RNNを使わず並列処理が可能です。自己注意機構により長距離依存関係を効率的に捉えられます。",
        "rejected": "Transformerは新しいAIモデルです。いろいろなことができます。",
    },
    {
        "topic": "要約",
        "question": "要約：スマートシティは都市インフラにIoT・AI・ビッグデータを活用して交通・エネルギー・防災を最適化します。",
        "chosen":   "スマートシティはIoT・AI・ビッグデータで都市インフラの交通・エネルギー・防災を最適化します。",
        "rejected": "スマートシティはITを使った街です。便利です。",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：ゲノム解析技術の進歩により個別化医療の実現が近づいています。",
        "chosen":   "Advances in genomic analysis technology are bringing personalized medicine closer to reality.",
        "rejected": "DNA analysis helps medicine.",
    },
    {
        "topic": "創作",
        "question": "夜明けの海を見ている老漁師の場面を書いてください。",
        "chosen":   "水平線が赤く染まる前に、三郎はすでに網を降ろしていた。五十年、この海と生きてきた。若い頃は魚を求めて沖に出たが、今は海が呼ぶから来る。それだけだ。",
        "rejected": "老漁師が海にいます。毎日魚を取ります。海が好きです。",
    },
    {
        "topic": "質問応答",
        "question": "RAGとは何ですか？",
        "chosen":   "RAG（Retrieval-Augmented Generation）は外部知識ベースから関連文書を検索し、それをコンテキストとしてLLMに提供して回答を生成する手法です。事実性向上とハルシネーション抑制に有効です。",
        "rejected": "RAGはAI技術の一つです。情報を検索します。",
    },
    {
        "topic": "要約",
        "question": "要約：NFTは非代替性トークンと呼ばれるブロックチェーン上のデジタル証明書で、アート・ゲーム・音楽のデジタル資産の所有権を証明します。",
        "chosen":   "NFTはブロックチェーン上のデジタル証明書で、アート・ゲーム・音楽などデジタル資産の所有権を証明します。",
        "rejected": "NFTはデジタルのものです。ブロックチェーンに関係あります。",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：The development of mRNA technology has opened new possibilities in vaccine design.",
        "chosen":   "mRNA技術の発展により、ワクチン設計における新たな可能性が開かれました。",
        "rejected": "mRNAのワクチン技術が進歩しました。",
    },
    {
        "topic": "創作",
        "question": "森の中で道に迷った子どもの場面を書いてください。",
        "chosen":   "木の葉が重なって空を隠し、どちらから来たのかわからなくなった。ハルは立ち止まり、耳を澄ました。川の音——右から聞こえる。川があれば道がある。父が教えてくれた言葉が足を動かした。",
        "rejected": "子どもが森で迷っています。怖いです。助けが来ます。",
    },
    {
        "topic": "質問応答",
        "question": "LoRAとはどのような手法ですか？",
        "chosen":   "LoRA（Low-Rank Adaptation）は大規模モデルを少ないパラメータで効率的にファインチューニングする手法です。重み行列の変化を低ランク行列の積で近似し、元の重みを固定したまま少数の追加パラメータのみを学習します。",
        "rejected": "LoRAはモデルを調整する技術です。効率的です。",
    },
    {
        "topic": "要約",
        "question": "要約：ウェアラブルデバイスの普及により、心拍・血圧・睡眠などのバイタルデータを常時計測し、AIが健康リスクを予測する予防医療が現実のものとなっています。",
        "chosen":   "ウェアラブルデバイスでバイタルデータを常時計測し、AIが健康リスクを予測する予防医療が実現しています。",
        "rejected": "スマートウォッチで体の状態がわかります。健康に良いです。",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：自動翻訳の精度は年々向上しており、専門家の翻訳に匹敵する場合もあります。",
        "chosen":   "The accuracy of machine translation is improving year by year, sometimes rivaling that of professional translators.",
        "rejected": "Machine translation is getting better every year.",
    },
    {
        "topic": "創作",
        "question": "宇宙船の最後の通信を描いてください。",
        "chosen":   "「地球、聞こえますか——」電流が弱まる中、船長の声は静かだった。「こちらは美しい星雲の中にいます。恐怖はありません。ただ……」ノイズが走り、信号は途絶えた。管制室では誰も何も言えなかった。",
        "rejected": "宇宙船が最後の通信をします。信号が切れます。悲しいです。",
    },
    {
        "topic": "質問応答",
        "question": "フラッシュアテンションとは何ですか？",
        "chosen":   "FlashAttentionはGPUのHBMへのアクセス回数を削減するためにタイリング手法でAttentionを計算する高速実装です。メモリ帯域ボトルネックを解消し、長シーケンスの処理速度とメモリ効率を大幅に改善します。",
        "rejected": "FlashAttentionはAttentionを速く計算する方法です。",
    },
    {
        "topic": "要約",
        "question": "要約：デジタルツインとは物理的な設備・工場・都市の精密なデジタルレプリカで、シミュレーションを通じて設計最適化・故障予測・意思決定支援を行う技術です。",
        "chosen":   "デジタルツインは設備・工場・都市の精密なデジタル複製で、シミュレーションによる設計最適化・故障予測・意思決定支援を実現します。",
        "rejected": "デジタルツインはデジタルのコピーです。シミュレーションができます。",
    },
    {
        "topic": "翻訳",
        "question": "日本語に翻訳：The convergence of AI and robotics is enabling autonomous systems to perform complex physical tasks.",
        "chosen":   "AIとロボティクスの融合により、自律システムが複雑な物理タスクを実行できるようになっています。",
        "rejected": "AIとロボットが一緒になりました。いろいろなことができます。",
    },
    {
        "topic": "創作",
        "question": "人口が激減した未来の村を描写してください。",
        "chosen":   "商店街には8軒の建物があり、営業しているのは3軒だった。郵便局の窓口は週2回しか開かない。それでも田中老人は毎朝6時に神社の鐘をつく。「誰も来なくても、鐘は鳴らさないといけない」",
        "rejected": "村に人が少ないです。お店も少ないです。寂しいです。",
    },
    {
        "topic": "質問応答",
        "question": "ZK証明（ゼロ知識証明）とは何ですか？",
        "chosen":   "ゼロ知識証明は証明者が検証者に情報の正しさを、その情報自体を開示せずに証明できる暗号プロトコルです。プライバシー保護・ブロックチェーンのスケーリング（zk-Rollup）に活用されています。",
        "rejected": "ゼロ知識証明は秘密を守りながら証明する方法です。暗号に使います。",
    },
    {
        "topic": "要約",
        "question": "要約：サステナブルファッションは環境負荷の低い素材を使い、公正な労働環境で生産し、循環型のビジネスモデルを採用するファッション産業の新潮流です。",
        "chosen":   "サステナブルファッションは環境低負荷素材・公正労働・循環型ビジネスモデルを組み合わせたファッション産業の新潮流です。",
        "rejected": "環境に優しいファッションです。良いことです。",
    },
    {
        "topic": "翻訳",
        "question": "英語に翻訳：大阪・関西万博は2025年に開催され、未来社会のデザインをテーマとしています。",
        "chosen":   "The Osaka-Kansai Expo, held in 2025, is themed around the design of a future society.",
        "rejected": "Osaka has an expo in 2025 about the future.",
    },
    {
        "topic": "創作",
        "question": "最後の本屋を守る老女の場面を書いてください。",
        "chosen":   "街に本屋はここだけになった。澄子は毎朝シャッターを上げながら、父が残した棚の並びを変えない。電子書籍の時代に、それでもドアを開けるたびに誰かが来る。彼らは本を探しているのではなく、紙の匂いを探している。",
        "rejected": "本屋のおばあさんが店を守っています。本が好きです。頑張っています。",
    },
]

assert len(PAIRS_TRAIN) == 80, f"Expected 80 train pairs, got {len(PAIRS_TRAIN)}"
assert len(PAIRS_VAL)   == 20, f"Expected 20 val pairs, got {len(PAIRS_VAL)}"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.pairs)

    def _encode(self, question, response):
        text = question + "[SEP]" + response
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        pair   = self.pairs[idx]
        chosen  = self._encode(pair["question"], pair["chosen"])
        rejected = self._encode(pair["question"], pair["rejected"])
        return {
            "chosen_input_ids":      chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids":      rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.bert   = AutoModel.from_pretrained(model_id)
        hidden_size = self.bert.config.hidden_size
        self.head   = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        return self.head(cls_hidden).squeeze(-1)           # (B,)


def bradley_terry_loss(r_chosen, r_rejected):
    return -torch.mean(torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-9))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reward_model(model, train_loader, val_loader, epochs=5, lr=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(DEVICE)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            c_ids  = batch["chosen_input_ids"].to(DEVICE)
            c_mask = batch["chosen_attention_mask"].to(DEVICE)
            r_ids  = batch["rejected_input_ids"].to(DEVICE)
            r_mask = batch["rejected_attention_mask"].to(DEVICE)

            r_chosen   = model(c_ids,  c_mask)
            r_rejected = model(r_ids,  r_mask)
            loss       = bradley_terry_loss(r_chosen, r_rejected)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation accuracy (how often chosen > rejected)
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for batch in val_loader:
                c_ids  = batch["chosen_input_ids"].to(DEVICE)
                c_mask = batch["chosen_attention_mask"].to(DEVICE)
                r_ids  = batch["rejected_input_ids"].to(DEVICE)
                r_mask = batch["rejected_attention_mask"].to(DEVICE)
                r_c    = model(c_ids, c_mask)
                r_r    = model(r_ids, r_mask)
                correct += (r_c > r_r).sum().item()
                total   += len(r_c)
        val_acc = correct / max(total, 1)

        print(f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
              f"val_acc={val_acc:.3f}")
        history.append({"epoch": epoch + 1, "loss": avg_loss, "val_acc": val_acc})

    return history


# ---------------------------------------------------------------------------
# Evaluation on new responses
# ---------------------------------------------------------------------------

def evaluate_responses(model, tokenizer, good_responses, bad_responses,
                        max_length=128):
    """Score 10 good + 10 bad responses; return raw scores and AUC."""
    model.eval()
    scores  = []
    labels  = []

    def score_text(text):
        enc = tokenizer(
            text, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt"
        )
        with torch.no_grad():
            s = model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            ).item()
        return s

    for r in good_responses:
        scores.append(score_text(r))
        labels.append(1)
    for r in bad_responses:
        scores.append(score_text(r))
        labels.append(0)

    auc           = roc_auc_score(labels, scores)
    good_mean     = float(np.mean(scores[:len(good_responses)]))
    bad_mean      = float(np.mean(scores[len(good_responses):]))
    reward_gap    = good_mean - bad_mean
    return scores, labels, auc, good_mean, bad_mean, reward_gap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RLHF Reward Model Training")
    print(f"  device : {DEVICE}")
    print(f"  model  : {MODEL_ID}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    train_ds     = PreferenceDataset(PAIRS_TRAIN, tokenizer)
    val_ds       = PreferenceDataset(PAIRS_VAL,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,
                              num_workers=0)

    print(f"\nTrain pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")

    model = RewardModel(MODEL_ID)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    print("\nTraining ...")
    history = train_reward_model(
        model, train_loader, val_loader, epochs=5, lr=1e-5
    )

    # ── Evaluate on 20 new responses (10 good + 10 bad) ─────────────────
    GOOD_RESPONSES = [
        "量子もつれは二つの粒子が空間的に離れていても互いの状態が瞬時に相関する量子力学的現象です。アインシュタインは「不気味な遠隔作用」と呼び懐疑的でしたが、ベルの不等式の実験的検証により実在が確認されています。",
        "BERT（Bidirectional Encoder Representations from Transformers）は双方向のAttentionで文脈を捉えるエンコーダモデルです。MLMとNSPで事前学習し、各種下流タスクへのファインチューニングで高精度を達成しました。",
        "日本の社会保障制度は医療・年金・介護・雇用の四本柱から成り、社会保険料と税財源で運営されています。少子高齢化による給付増と保険料収入減の構造的課題に対応するため、給付と負担の見直しが継続的に議論されています。",
        "GDPは一定期間内に国内で生産された付加価値の合計で、消費・投資・政府支出・純輸出の合計として計算されます。実質GDPはインフレの影響を除いた経済成長の実態を反映します。",
        "機械学習の過学習は、モデルが訓練データに過度に適合し汎化性能が低下する問題です。Dropout・L1/L2正則化・データ拡張・アーリーストッピング・クロスバリデーションが主な対策として使われます。",
        "DNA複製では二重らせんが解けてそれぞれを鋳型にDNAポリメラーゼが相補的な塩基を合成します。複製は半保存的で、親鎖と新規合成鎖が各娘細胞に分配されます。岡崎フラグメントにより遅れ鎖が合成されます。",
        "太陽光発電は光起電力効果を利用しシリコンp-n接合に光を当て起電力を発生させます。変換効率は市販品で20%前後、研究段階の多接合型で40%以上を達成しています。",
        "TCP/IPはインターネット通信の基盤プロトコルです。IPが送受信アドレスを管理しパケットをルーティングし、TCPが信頼性のある順序付きデータ転送を保証します。UDPはTCPより軽量で低遅延が必要な用途に使われます。",
        "強化学習のPPO（Proximal Policy Optimization）はクリッピングにより方策更新量を制約しながら方策勾配を安定的に最適化するアルゴリズムです。サンプル効率と安定性のバランスが良くRLHFで広く使われています。",
        "原子炉は核分裂連鎖反応を制御してエネルギーを取り出す装置です。ウラン235の核分裂で放出された中性子が制御棒と減速材で調整されます。PWR（加圧水型）とBWR（沸騰水型）が主流の商用炉形式です。",
    ]
    BAD_RESPONSES = [
        "量子もつれは量子の不思議な現象です。科学者が研究しています。",
        "BERTはAIのモデルです。自然言語処理に使います。",
        "日本の社会保障は大切です。お金がかかります。",
        "GDPは国の経済の大きさを表します。大きいほど良いです。",
        "機械学習では過学習に注意が必要です。対策があります。",
        "DNAは二重らせん構造です。複製されます。",
        "太陽光発電は環境に優しいです。光を電気にします。",
        "インターネットはプロトコルで動いています。通信できます。",
        "PPOは強化学習のアルゴリズムです。良い手法です。",
        "原子炉は核分裂を使って電気を作ります。",
    ]

    print("\nEvaluating 20 new responses (10 good + 10 bad) ...")
    scores, labels, auc, good_mean, bad_mean, reward_gap = evaluate_responses(
        model, tokenizer, GOOD_RESPONSES, BAD_RESPONSES
    )

    print(f"\n=== Evaluation Results ===")
    print(f"  AUC                  : {auc:.4f}")
    print(f"  Good response reward : {good_mean:.4f}")
    print(f"  Bad  response reward : {bad_mean:.4f}")
    print(f"  Reward gap           : {reward_gap:.4f}")

    print("\nPer-response scores:")
    for i, (s, l) in enumerate(zip(scores, labels)):
        kind = "good" if l == 1 else "bad "
        print(f"  [{kind}] {s:+.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "train_pairs": len(PAIRS_TRAIN),
        "val_pairs": len(PAIRS_VAL),
        "epochs": 5,
        "learning_rate": 1e-5,
        "batch_size": 8,
        "loss_function": "Bradley-Terry: -mean(log(sigmoid(r_chosen - r_rejected)))",
        "training_history": history,
        "evaluation": {
            "n_good": len(GOOD_RESPONSES),
            "n_bad": len(BAD_RESPONSES),
            "auc": round(auc, 4),
            "good_response_mean_reward": round(good_mean, 4),
            "bad_response_mean_reward": round(bad_mean, 4),
            "reward_gap": round(reward_gap, 4),
            "scores": [round(s, 4) for s in scores],
            "labels": labels,
        },
    }

    out_path = os.path.join(_THIS_DIR, "reward_model_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
