import os
import subprocess
import requests
import boto3
import torch
from diffusers import DiffusionPipeline
import glob
import time
import dataclasses
import logging
from typing import Dict, Any, Union, Type
from aiohttp import web, ClientResponse

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler
from lib.server import start_server
from .data_types import InputData

# --- ロガーの設定 ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s[%(levelname)-5s] %(message)s")
log = logging.getLogger(__name__)

# --- グローバル設定 ---
BASE_MODEL_PATH = os.environ.get('BASE_MODEL_PATH', '/workspace/modelse/checkpointse/Illustriouse/solventeclipseVpred_v11.safetensors')
R2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# boto3クライアントは一度だけ初期化
s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)

# --- vast_runner.pyから移植した学習関数群 ---

def run_tagger(train_data_dir: str) -> str:
    log.info("--- Starting Tagger ---")
    model_dir = "/workspace/local_tagger_model"
    command = [
        "accelerate", "launch", "./sd-scripts/finetune/tag_images_by_wd14_tagger.py",
        train_data_dir, "--model_dir", model_dir, "--batch_size", "1",
        "--caption_extension", ".txt", "--general_threshold", "0.35",
        "--character_threshold", "0.85", "--onnx",
    ]
    log.info(f"Tagger Command: {' '.join(command)}")
    subprocess.run(command, check=True)
    log.info("--- Tagger Finished ---")
    caption_files = glob.glob(os.path.join(train_data_dir, "*.txt"))
    if caption_files:
        with open(caption_files[0], 'r') as f:
            return f.read()
    return ""

def run_training(job_id: str, train_data_dir: str) -> str:
    output_dir = f"/workspace/output/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "accelerate", "launch", "./sd-scripts/sdxl_train_network.py",
        "--pretrained_model_name_or_path", BASE_MODEL_PATH,
        "--train_data_dir", train_data_dir,
        "--output_dir", output_dir,
        "--output_name", f"{job_id}_lora",
        "--resolution", "1024,1024", "--train_batch_size", "1",
        "--max_train_epochs", "10", "--dataset_repeats", "30",
        "--save_every_n_epochs", "2", "--learning_rate", "1.0",
        "--unet_lr", "1.0", "--text_encoder_lr", "1.0",
        "--network_module", "networks.lora", "--network_dim", "64",
        "--network_alpha", "32", "--optimizer_type", "Prodigy",
        '--optimizer_args', 'decouple=True', 'weight_decay=0.01', 'use_bias_correction=True', 'd_coef=0.8', 'd0=5e-5', 'safeguard_warmup=True', 'betas=0.9,0.99',
        "--lr_scheduler", "cosine", "--mixed_precision", "bf16",
        "--save_precision", "bf16", "--gradient_checkpointing",
        "--xformers", "--no_half_vae", "--v_parameterization",
        "--min_snr_gamma", "5", "--save_model_as", "safetensors",
        "--caption_extension", ".txt", "--cache_latents", "--cache_latents_to_disk"
    ]
    log.info(f"--- Starting LoRA Training ---\nCommand: {' '.join(command)}")
    subprocess.run(command, check=True)
    log.info("--- LoRA Training Finished ---")
    lora_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    if not lora_files:
        raise FileNotFoundError("No LoRA file was generated.")
    return max(lora_files, key=os.path.getctime)

def generate_sample_image(lora_path: str, prompt: str, job_id: str) -> str:
    log.info("--- Loading pipeline for sample generation ---")
    pipeline = DiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16, custom_pipeline="lpw_stable_diffusion_xl"
    )
    pipeline.to("cuda")
    log.info(f"--- Loading LoRA weights from {lora_path} ---")
    pipeline.load_lora_weights(lora_path)
    log.info("--- Generating sample image ---")
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    sample_image_path = f"/workspace/sample_{job_id}.png"
    image.save(sample_image_path)
    log.info(f"--- Sample image saved to {sample_image_path} ---")
    return sample_image_path

def notify_backend(webhook_url: Optional[str], payload: Dict[str, Any]):
    if not webhook_url:
        log.info("Webhook URL not set. Skipping notification.")
        return
    try:
        requests.post(webhook_url, json=payload, timeout=10)
    except requests.RequestException as e:
        log.error(f"Failed to send webhook: {e}")

# --- PyWorker Endpoint Handler ---

@dataclasses.dataclass
class TrainLoraHandler(EndpointHandler[InputData]):

    @property
    def endpoint(self) -> str:
        return "/train-lora"

    @classmethod
    def payload_cls(cls) -> Type[InputData]:
        return InputData

    def generate_payload_json(self, payload: InputData) -> Dict[str, Any]:
        # このワーカーは他のモデルAPIを呼ばないので、実装は不要
        return {}

    def make_benchmark_payload(self) -> InputData:
        # ベンチマークはLoRA学習では行わないため、ダミーを返す
        return InputData.for_test()

    async def handle_request(self, payload: InputData) -> web.Response:
        """APIリクエストを受け取り、LoRA学習プロセス全体を実行する"""
        job_id = payload.job_id
        user_id = payload.user_id
        image_r2_key = payload.image_r2_key
        webhook_url = payload.webhook_url
        
        log.info(f"--- Received LoRA Training Job --- \nJobID: {job_id}, UserID: {user_id}, ImageKey: {image_r2_key}")

        local_image_path = f"/workspace/{os.path.basename(image_r2_key)}"
        
        try:
            # 1. R2から画像をダウンロード
            log.info(f"Downloading {image_r2_key} from R2 bucket {S3_BUCKET_NAME}...")
            s3_client.download_file(S3_BUCKET_NAME, image_r2_key, local_image_path)

            # 2. ディレクトリ構造を準備
            train_image_dir = f"/workspace/train_data/{job_id}/30_mychar"
            os.makedirs(train_image_dir, exist_ok=True)
            os.rename(local_image_path, os.path.join(train_image_dir, "image.jpg"))

            # 3. Taggerでキャプション生成
            caption = run_tagger(train_image_dir)
            log.info(f"Generated Caption: '{caption}'")

            # 4. LoRA学習実行
            lora_file_path = run_training(job_id, f"/workspace/train_data/{job_id}")

            # 5. LoRAをR2にアップロード
            lora_s3_key = f"users/{user_id}/artifacts/{job_id}/lora.safetensors"
            log.info(f"Uploading LoRA to r2://{S3_BUCKET_NAME}/{lora_s3_key}")
            s3_client.upload_file(lora_file_path, S3_BUCKET_NAME, lora_s3_key)

            # 6. サンプル画像生成
            sample_prompt = f"masterpiece, best quality, 1girl, {caption}"
            sample_image_path = generate_sample_image(lora_file_path, sample_prompt, job_id)

            # 7. サンプル画像をR2にアップロード
            sample_image_s3_key = f"users/{user_id}/artifacts/{job_id}/sample.png"
            log.info(f"Uploading sample image to r2://{S3_BUCKET_NAME}/{sample_image_s3_key}")
            s3_client.upload_file(sample_image_path, S3_BUCKET_NAME, sample_image_s3_key)

            # 8. 成功を通知 & レスポンス
            completion_payload = {
                "job_id": job_id, "status": "COMPLETED",
                "artifacts": {
                    "lora_url": f"r2://{S3_BUCKET_NAME}/{lora_s3_key}",
                    "sample_image_url": f"r2://{S3_BUCKET_NAME}/{sample_image_s3_key}"
                }
            }
            notify_backend(webhook_url, completion_payload)
            return web.json_response(completion_payload, status=200)

        except Exception as e:
            log.exception(f"An error occurred during LoRA training job {job_id}: {e}")
            error_payload = {"job_id": job_id, "status": "FAILED", "error": str(e)}
            notify_backend(webhook_url, error_payload)
            return web.json_response(error_payload, status=500)

    async def generate_client_response(self, client_request: web.Request, model_response: ClientResponse) -> Union[web.Response, web.StreamResponse]:
        # このハンドラは内部で完結するため、モデルサーバへのリクエスト(model_response)は発生しない。
        # handle_requestを直接呼び出すように`Backend`を少し改造する必要があるかもしれないが、
        # ここではPyWorkerの標準的な形式に合わせ、このメソッドは空にしておく。
        # 実際には、`backend.create_handler`が`handle_request`を呼び出す。
        pass

# --- サーバーの起動設定 ---

# このPyWorkerは他のモデルサーバーをラップするのではなく、自身が処理を実行する。
# そのため、`Backend`クラスのいくつかのパラメータは不要。
backend = Backend(
    # model_server_urlは使わないが、形式上必要
    model_server_url="", 
    # このワーカーは一度に一つの学習ジョブしか実行できない
    allow_parallel_requests=False,
    # ベンチマークは実行しない
    benchmark_handler=None,
)

# ルーティング設定
routes = [
    web.post("/train-lora", backend.create_handler(TrainLoraHandler())),
]

if __name__ == "__main__":
    # PyWorkerサーバーを起動
    start_server(backend, routes)
