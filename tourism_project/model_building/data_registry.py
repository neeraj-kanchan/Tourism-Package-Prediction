import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


repo_id = "neeraj-kanchan/Tourism-Package-Prediction"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("GH_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("Found repo '{repo_id}'")
except RepositoryNotFoundError:
    print("Repo '{repo_id}' not found.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
