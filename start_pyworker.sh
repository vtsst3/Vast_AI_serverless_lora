#!/bin/bash

# このスクリプトはVast.aiの 'On-start script' として設定します。
# 役割は、コンテナが起動する前に必要なファイル（例：ベースモデル）を準備することです。

echo "--- [On-start script] Preparing environment... ---"

# ベースモデルのパスとURLを設定
# MODEL_URLは実際のダウンロードURLに書き換えてください。
MODEL_PATH=$(dirname "$BASE_MODEL_PATH")
MODEL_URL="https://civitai.com/api/download/models/306497" # 仮のURL。正しいURLに要変更
MODEL_DIR=$(dirname "$BASE_MODEL_PATH")

# 環境変数BASE_MODEL_PATHが設定されているか確認
if [ -z "$BASE_MODEL_PATH" ]; then
    echo "Error: BASE_MODEL_PATH environment variable is not set."
    exit 1
fi

# モデルファイルが存在しない場合にのみダウンロード
if [ ! -f "$BASE_MODEL_PATH" ]; then
    echo "Base model not found at $BASE_MODEL_PATH. Downloading..."
    # ディレクトリが存在しない場合は作成
    mkdir -p "$MODEL_DIR"
    # aria2cを使用して高速ダウンロード
    aria2c -x 16 -s 16 -k 1M -d "$MODEL_DIR" -o "$(basename "$BASE_MODEL_PATH")" "$MODEL_URL"
    if [ $? -ne 0 ]; then
        echo "Failed to download the base model. Exiting."
        exit 1
    fi
    echo "Base model downloaded successfully."
else
    echo "Base model already exists at $BASE_MODEL_PATH. Skipping download."
fi

echo "--- [On-start script] Environment preparation finished. ---"
# このスクリプトの終了後、DockerfileのCMDで定義されたPyWorkerが自動的に起動します。
