"""Upload trained checkpoints to the HF model repo (private until release).

Usage: python scripts/publish_checkpoints.py TOK_CKPT WM_CKPT [REPO]
"""
import sys

from huggingface_hub import HfApi

repo = sys.argv[3] if len(sys.argv) > 3 else "kevin510/nano-world-model"
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
api.upload_file(path_or_fileobj=sys.argv[1], path_in_repo="tokenizer.pt", repo_id=repo)
api.upload_file(path_or_fileobj=sys.argv[2], path_in_repo="world_model.pt", repo_id=repo)
print(f"published tokenizer.pt + world_model.pt to {repo}")
