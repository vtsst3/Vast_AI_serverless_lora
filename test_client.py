import asyncio
import os
import uuid
from vastai import Serverless

# --- 設定 ---
# 環境変数からVast.aiのAPIキーとエンドポイント名を取得
# 事前に `export VAST_API_KEY="..."` と `export ENDPOINT_NAME="..."` を設定してください。
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "my-lora-trainer-endpoint")

async def main():
    """
    Vast.aiサーバーレスエンドポイントにLoRA学習ジョブを投入するクライアント。
    """
    print(f"--- Starting LoRA training job on endpoint: {ENDPOINT_NAME} ---")

    # ユニークなジョブIDを生成
    job_id = str(uuid.uuid4())
    
    # APIに送信するペイロードを定義
    # この構造は pyworker/workers/lora_trainer/data_types.py の InputData と一致させる
    job_payload = {
        "user_id": "local_test_user",
        "job_id": job_id,
        "image_r2_key": "training_images/my_character.jpg", # R2にアップロード済みの学習用画像のパス
        "webhook_url": "https://example.com/my-webhook-endpoint" # (任意)
    }

    try:
        async with Serverless() as client:
            # エンドポイントを取得
            endpoint = await client.get_endpoint(name=ENDPOINT_NAME)

            print(f"Sending job (ID: {job_id}) to worker...")
            
            # /train-lora エンドポイントにリクエストを送信
            # `cost` はスケーリングのヒント。`count_workload` で定義した値に合わせる。
            response = await endpoint.request("/train-lora", {"input": job_payload}, cost=100.0)

            # サーバーからのレスポンスを出力
            print("\n--- Job Response ---")
            print(response)
            print("--------------------\n")

            if response.get("response", {}).get("status") == "COMPLETED":
                print("✅ Job completed successfully!")
            else:
                print("❌ Job failed or is processing. Check logs for details.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure that `VAST_API_KEY` and `ENDPOINT_NAME` environment variables are set correctly.")
        print("And that the endpoint is active on Vast.ai.")

if __name__ == "__main__":
    # vastai_sdkをインストールする必要があります: pip install vastai_sdk
    asyncio.run(main())
