# 深層強化学習Summer2025最終課題

## 概要
- 「Gymnasium Carracing-v3」上でPPOを実装しました。
- ハイパーパラメータや報酬関数の調整により、完璧な走りではないものの、スコア900超え（コースを1周完走できる程度の性能）のモデルを作成することに成功しました。
- 学習済みエージェントが画像のどこに注目して行動を決定しているのかを、**Grad-CAM**を用いて可視化し
- 学習の安定化を目指して、価値関数のclippingや報酬の正規化、学習率のアニーリングなどにも取り組みました。

## 動作環境

* Python 3.10
* PyTorch
* Gymnasium

詳細なライブラリは`requirements.txt`を参照してください。

## セットアップ

1.  リポジトリをクローンします。
    ```bash
    git clone https://github.com/lumenyuto/RLS2025-final-assignment
    cd RLS2025-final-assignment
    ```

2.  必要なライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

## 使い方

### 1. Agentの学習

以下のコマンドで学習を開始します。学習済みモデルは`model_ppo`ディレクトリに保存されます。
- 報酬のclipping無し

```bash
python main.py --mode train 
```
- 報酬のclipping有り
```bash
python main.py --mode train --value_clipping
```


### 2. 学習済みエージェントのデモ

学習済みの重みを使って、エージェントの走行を動画で確認します。
- 報酬関数のclipping無し
```bash
python main.py --mode demo --model_dir best_ppo_1
```

- 報酬関数のclipping有り
```bash
python main.py --mode demo --model_dir best_ppo_2
```

## 学習結果
### 1.学習曲線（報酬関数のclipping無し）
![学習曲線](https://github.com/lumenyuto/RLs2025-final-assignment/blob/main/images/best_ppo_1.png?raw=true)

### 2.学習曲線（報酬関数のclipping有り）
![学習曲線](https://github.com/lumenyuto/RLs2025-final-assignment/blob/main/images/best_ppo_2.png?raw=true)

### エージェントの走行動画


## 考察




