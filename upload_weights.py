"""Upload ./artifacts to a Hugging Face model repo (weights must NOT be in the GitHub repo).

    huggingface-cli login            # paste a token with "write" permission
    python upload_weights.py --repo YOUR_HF_USERNAME/swe015-chat-moderation

Afterwards anyone can run the app without training:
    MODEL_REPO=YOUR_HF_USERNAME/swe015-chat-moderation python app.py
"""
import argparse

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--art", default="artifacts")
args = ap.parse_args()
api = HfApi()
api.create_repo(args.repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=args.art, repo_id=args.repo, repo_type="model")
print(f"Uploaded. Link for the README / slides: https://huggingface.co/{args.repo}")
