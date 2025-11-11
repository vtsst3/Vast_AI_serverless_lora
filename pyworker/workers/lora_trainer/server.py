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
from typing import Dict, Any, Optional
from aiohttp import web
import json
import asyncio

# --- ロガーの設定 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(message)s")
log = logging.getLogger(__name__)

# --- データクラスの定義 ---
@dataclasses.dataclass
class InputData:
    job_id: str
    user_id: str
    image_r2_key: str
    webhook_url: Optional[str] = None

# --- グローバル設定 ---
BASE_MODEL_PATH = os.environ.get('BASE_MODEL_PATH', '/workspace/modelse/checkpointse/Illustriouse/solventeclipseVpred_v11.safetensors')
R2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
s3_client = None

def initialize_s3_client():
    global s3_client
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not all([R2_ENDPOINT_URL, S3_BUCKET_NAME, access_key, secret_key]):
        log.warning("S3/R2 environment variables are not fully set. S3 operations will be skipped.")
        s3_client = None
        return
    s3_client = boto3.client('s3', endpoint_url=R2_ENDPOINT_URL, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    log.info("S3/R2 client initialized successfully.")

# --- 学習関数群 ---
def run_command(command: list, cwd: str):
    log.info(f"Running command: {' '.join(command)} in {cwd}")
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in iter(process.stdout.readline, ''):
        log.info(line.strip())
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def run_tagger(train_data_dir: str) -> str:
    log.info("--- Starting Tagger ---")
    command = [
        "accelerate", "launch", "./sd-scripts/finetune/tag_images_by_wd14_tagger.py",
        train_data_dir, "--model_dir=/workspace/local_tagger_model", "--batch_size=1",
        "--caption_extension=.txt", "--general_threshold=0.35", "--character_threshold=0.85", "--onnx",
    ]
    run_command(command, cwd="/workspace/lora-worker")
    log.info("--- Tagger Finished ---")
    caption_files = glob.glob(os.path.join(train_data_dir, "*.txt"))
    return open(caption_files[0], 'r', encoding='utf-8').read() if caption_files else ""

def run_training(job_id: str, train_data_dir: str) -> str:
    output_dir = f"/workspace/output/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "accelerate", "launch", "./sd-scripts/sdxl_train_network.py",
        "--pretrained_model_name_or_path", BASE_MODEL_PATH, "--train_data_dir", train_data_dir,
        "--output_dir", output_dir, "--output_name", f"{job_id}_lora",
        "--resolution=1024,1024", "--train_batch_size=1", "--max_train_epochs=10",
        "--dataset_repeats=30", "--save_every_n_epochs=2", "--learning_rate=1.0",
        "--unet_lr=1.0", "--text_encoder_lr=1.0", "--network_module=networks.lora",
        "--network_dim=64", "--network_alpha=32", "--optimizer_type=Prodigy",
        '--optimizer_args', 'decouple=True', 'weight_decay=0.01', 'use_bias_correction=True', 'd_coef=0.8', 'd0=5e-5', 'safeguard_warmup=True', 'betas=0.9,0.99',
        "--lr_scheduler=cosine", "--mixed_precision=bf16", "--save_precision=bf16",
        "--gradient_checkpointing", "--xformers", "--no_half_vae", "--v_parameterization",
        "--min_snr_gamma=5", "--save_model_as=safetensors", "--caption_extension=.txt",
        "--cache_latents", "--cache_latents_to_disk"
    ]
    run_command(command, cwd="/workspace/lora-worker")
    log.info("--- LoRA Training Finished ---")
    lora_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    if not lora_files: raise FileNotFoundError("No LoRA file was generated.")
    return max(lora_files, key=os.path.getctime)

def generate_sample_image(lora_path: str, prompt: str, job_id: str) -> str:
    log.info("--- Loading pipeline for sample generation ---")
    pipeline = DiffusionPipeline.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, custom_pipeline="lpw_stable_diffusion_xl")
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

def run_lora_training_process(payload: InputData):
    job_id = payload.job_id
    try:
        log.info(f"--- Starting LoRA Training Job --- \nJobID: {job_id}")
        local_image_path = f"/workspace/{os.path.basename(payload.image_r2_key)}"
        
        if s3_client:
            log.info(f"Downloading {payload.image_r2_key} from R2...")
            s3_client.download_file(S3_BUCKET_NAME, payload.image_r2_key, local_image_path)
        else:
            raise RuntimeError("S3 client is not initialized. Cannot download image.")

        train_image_dir = f"/workspace/train_data/{job_id}/30_mychar"
        os.makedirs(train_image_dir, exist_ok=True)
        os.rename(local_image_path, os.path.join(train_image_dir, "image.jpg"))

        caption = run_tagger(train_image_dir)
        log.info(f"Generated Caption: '{caption}'")

        lora_file_path = run_training(job_id, f"/workspace/train_data/{job_id}")

        lora_s3_key = f"users/{payload.user_id}/artifacts/{job_id}/lora.safetensors"
        if s3_client:
            log.info(f"Uploading LoRA to r2://{S3_BUCKET_NAME}/{lora_s3_key}")
            s3_client.upload_file(lora_file_path, S3_BUCKET_NAME, lora_s3_key)

        sample_prompt = f"masterpiece, best quality, 1girl, {caption}"
        sample_image_path = generate_sample_image(lora_file_path, sample_prompt, job_id)

        sample_image_s3_key = f"users/{payload.user_id}/artifacts/{job_id}/sample.png"
        if s3_client:
            log.info(f"Uploading sample image to r2://{S3_BUCKET_NAME}/{sample_image_s3_key}")
            s3_client.upload_file(sample_image_path, S3_BUCKET_NAME, sample_image_s3_key)

        completion_payload = {
            "job_id": job_id, "status": "COMPLETED",
            "artifacts": {"lora_url": f"r2://{S3_BUCKET_NAME}/{lora_s3_key}", "sample_image_url": f"r2://{S3_BUCKET_NAME}/{sample_image_s3_key}"}
        }
        notify_backend(payload.webhook_url, completion_payload)
        log.info(f"Job {job_id} completed successfully.")
    except Exception as e:
        log.exception(f"An error occurred during LoRA training job {job_id}: {e}")
        error_payload = {"job_id": job_id, "status": "FAILED", "error": str(e)}
        notify_backend(payload.webhook_url, error_payload)

async def handle_train_lora(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        payload = InputData(**data)
    except (json.JSONDecodeError, TypeError) as e:
        log.error(f"Invalid JSON payload: {e}")
        return web.json_response({"status": "FAILED", "error": f"Invalid JSON payload: {e}"}, status=400)

    log.info(f"Accepted job {payload.job_id}. Starting training process in background.")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_lora_training_process, payload)
    return web.json_response({"status": "ACCEPTED", "job_id": payload.job_id}, status=202)

async def main():
    initialize_s3_client()
    app = web.Application()
    app.add_routes([web.post('/train-lora', handle_train_lora)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8000)
    log.info("======== LoRA Worker Server starting on 0.0.0.0:8000 ========")
    await site.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server shutting down.")
