import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

def download_hf_subset(repo_id: str, local_dir: str, num_files: int = 5):
    """Downloads a small subset of files from a HuggingFace dataset repository."""
    os.makedirs(local_dir, exist_ok=True)
    logger.info(f"Connecting to HuggingFace Repo: {repo_id}")
    
    try:
        # Get list of files in the repo
        files = list_repo_files(repo_id, repo_type="dataset")
        
        # Filter for actual image files (e.g., .tif)
        tif_files = [f for f in files if f.endswith('.tif')]
        
        if not tif_files:
            logger.warning("No .tif files found in the repository root. Trying to download first available files.")
            tif_files = [f for f in files if not f.startswith('.')]
            
        subset = tif_files[:num_files]
        logger.info(f"Found {len(tif_files)} files. Downloading subset of {len(subset)} files...")
        
        downloaded = []
        for file in subset:
            logger.info(f"Downloading {file}...")
            # We download directly into the specific folder
            path = hf_hub_download(
                repo_id=repo_id, 
                repo_type="dataset", 
                filename=file, 
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            downloaded.append(path)
            
        logger.info(f"Successfully downloaded {len(downloaded)} files to {local_dir}")
        return downloaded
        
    except Exception as e:
        logger.error(f"Failed to download from HuggingFace: {e}")
        logger.info("Note: Some datasets require you to accept terms on HuggingFace and provide a HF_TOKEN environment variable.")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=str, default='ibm-nasa-geospatial/hls_burn_scars')
    parser.add_argument('--dir', type=str, default='../datasets/hls_burn_scars/train')
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()
    
    # Resolve relative path
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.dir))
    download_hf_subset(args.repo, out_dir, args.count)
