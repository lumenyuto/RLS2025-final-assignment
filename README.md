# 深層強化学習Summer2025最終課題

## 概要

このプロジェクトでは、深層強化学習アルゴリズム **PPO (Proximal Policy Optimization)** を用いて、Gymnasiumの**CarRacing-v3**環境を攻略するエージェントを学習させます。

さらに、学習済みエージェントが画像の**どこに注目して**行動を決定しているのかを、**Grad-CAM**を用いて可視化し、その判断根拠を分析します。
## 概要
- 「Gymnasium Carracing-v3」上でPPOを実装しました。
- ハイパーパラメータや報酬関数の調整により、完璧な走りではないものの、スコア900超え（コースを1周完走できる程度の性能）のモデルを作成することに成功しました。
- 学習の安定化を目指して、価値関数のclippingや報酬の正規化、学習率のアニーリングなどにも取り組みました。

## 動作環境

* Python 3.10
* PyTorch
* Gymnasium

詳細なライブラリは`requirements.txt`を参照してください。

## セットアップ

1.  リポジトリをクローンします。
    ```bash
    git clone [https://github.com/your-username/PPO-CarRacing-GradCAM.git](https://github.com/your-username/PPO-CarRacing-GradCAM.git)
    cd PPO-CarRacing-GradCAM
    ```

2.  必要なライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

## 使い方

### 1. エージェントの学習

以下のコマンドで学習を開始します。学習済みモデルは`model_ppo`ディレクトリに保存されます。

```bash
python main.py --mode train
```

### 2. 学習済みエージェントのデモ

学習済みの重みを使って、エージェントの走行を動画で確認します。

```bash
python main.py --mode demo --model_dir 'model_ppo'
```

### 3. Grad-CAMによる可視化

エージェントの判断根拠をGrad-CAMで可視化します。

```bash
python visualize_cam.py --model_dir 'model_ppo'
```

## 学習結果

### 学習曲線

（ここに`trainer.plot()`で生成した学習曲線の画像を貼り付け）

### エージェントの走行動画

（ここに`trainer.visualize()`で生成した動画のGIFを貼り付け）

## 考察：エージェントはどこを見ているか？

Grad-CAMによる可視化結果から、エージェントは以下のような特徴を持っていることが示唆されました。

* 



