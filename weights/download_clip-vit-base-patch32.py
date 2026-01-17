# weights/download_clip_vit_base-patch32.py
import os
from transformers import CLIPModel, CLIPProcessor


def download_and_save():
    # 1. Set target save path (absolute path)
    # Get current script directory (i.e. weights/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Target folder: weights/models/clip-vit-base-patch32
    save_directory = os.path.join(current_dir, 'models', 'clip-vit-base-patch32')

    print(f"Preparing to download model...")
    print(f"Source model ID: openai/clip-vit-base-patch32")
    print(f"Target save path: {save_directory}")

    # 2. Specify HuggingFace official model ID
    model_id = "openai/clip-vit-base-patch32"

    try:
        print("\nConnecting to HuggingFace to download (please ensure network is stable)...")

        # A. Download from cloud and load to memory
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)

        print("Model download completed, writing to local disk...")

        # B. Save in-memory model to specified directory
        model.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)

        print(f"\nSuccess! Model saved to: {save_directory}")
        print("Folder should contain: pytorch_model.bin, config.json, preprocessor_config.json, etc.")

    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Hint: Please check your network connection. If unable to connect to HuggingFace, you may need to enable VPN.")


if __name__ == '__main__':
    download_and_save()