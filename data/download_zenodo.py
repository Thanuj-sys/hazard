import os
import requests
import argparse
from loguru import logger

def download_zenodo_subset(record_id: str, local_dir: str, num_files: int = 5):
    """Downloads a small subset of files from a Zenodo record."""
    os.makedirs(local_dir, exist_ok=True)
    
    api_url = f"https://zenodo.org/api/records/{record_id}"
    logger.info(f"Connecting to Zenodo API: {api_url}")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        files = data.get('files', [])
        if not files:
            logger.warning("No files found in the Zenodo record.")
            return []
            
        # Prioritize image/tif files if possible, otherwise grab first N
        tif_files = [f for f in files if f['key'].endswith('.tif') or f['key'].endswith('.zip')]
        if not tif_files:
            tif_files = files
            
        subset = tif_files[:num_files]
        logger.info(f"Found {len(files)} files. Downloading subset of {len(subset)} files...")
        
        downloaded = []
        for file_info in subset:
            filename = file_info['key']
            download_url = file_info['links']['self']
            
            output_path = os.path.join(local_dir, filename)
            logger.info(f"Downloading {filename} ({file_info['size'] / 1024 / 1024:.2f} MB)...")
            
            # Streaming download
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            downloaded.append(output_path)
            logger.info(f"Saved to {output_path}")
            
        return downloaded
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to communicate with Zenodo API: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 11278238 is the Global S1S2-Water Benchmark Dataset
    parser.add_argument('--record', type=str, default='11278238')
    parser.add_argument('--dir', type=str, default='../datasets/s1s2_water/train')
    parser.add_argument('--count', type=int, default=2) # Keep count low to avoid massive downloads
    args = parser.parse_args()
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.dir))
    download_zenodo_subset(args.record, out_dir, args.count)
