import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_sample_tiff(filepath, num_bands, height, width, dtype, scale=1.0):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    transform = from_origin(0, 0, 10, 10) # Dummy transform
    
    with rasterio.open(
        filepath, 'w',
        driver='GTiff',
        height=height, width=width,
        count=num_bands,
        dtype=dtype,
        crs='+proj=latlong',
        transform=transform,
        tiled=True, # COG format
        compress='lzw'
    ) as dst:
        for i in range(1, num_bands + 1):
            data = (np.random.rand(height, width) * scale).astype(dtype)
            dst.write(data, i)
    print(f"Created {filepath}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    
    print("Generating sample datasets based on specifications...")
    
    # 1. S1S2-Water Dataset
    # S1: 2 bands (VV, VH), int16 (dB * 100)
    # S2: 6 bands, uint16 (TOA * 10000)
    # Target: 512x512 crops (original is 10980x10980)
    s1s2_dir = os.path.join(base_dir, 's1s2_water', 'train')
    create_sample_tiff(os.path.join(s1s2_dir, 'sample_s1.tif'), 2, 512, 512, 'int16', 1000)
    create_sample_tiff(os.path.join(s1s2_dir, 'sample_s2.tif'), 6, 512, 512, 'uint16', 5000)
    create_sample_tiff(os.path.join(s1s2_dir, 'sample_mask.tif'), 1, 512, 512, 'uint8', 1)
    
    # 2. HLS Burn Scar Dataset
    # 6 bands + 1 mask, 512x512
    hls_dir = os.path.join(base_dir, 'hls_burn_scars', 'train')
    create_sample_tiff(os.path.join(hls_dir, 'sample_hls.tif'), 6, 512, 512, 'uint16', 5000)
    create_sample_tiff(os.path.join(hls_dir, 'sample_mask.tif'), 1, 512, 512, 'uint8', 1)
    
    # 3. CAS Landslide Dataset
    # RGB (3 bands) + mask, 512x512
    cas_dir = os.path.join(base_dir, 'cas_landslide', 'train')
    create_sample_tiff(os.path.join(cas_dir, 'sample_rgb.tif'), 3, 512, 512, 'uint8', 255)
    create_sample_tiff(os.path.join(cas_dir, 'sample_mask.tif'), 1, 512, 512, 'uint8', 1)

    print("\nSample generation complete. Datasets ready for pipeline testing.")

if __name__ == '__main__':
    main()
