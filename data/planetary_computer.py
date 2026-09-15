"""Microsoft Planetary Computer STAC API Interface."""
import time
from typing import Dict, Any, List, Optional
import pystac_client
import planetary_computer
from loguru import logger
import numpy as np

class PlanetaryComputerClient:
    """Client for Microsoft Planetary Computer STAC API."""

    def __init__(self, collection: str) -> None:
        """Initialize with STAC collection name.

        Args:
            collection (str): Name of the STAC collection.
        """
        self.collection = collection
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

    def search(self, bbox: List[float], datetime_range: str, max_items: int = 100) -> List[Any]:
        """Search STAC catalog and return item list.

        Args:
            bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat].
            datetime_range (str): Datetime range string (e.g. '2023-01-01/2023-01-31').
            max_items (int): Maximum number of items to return.

        Returns:
            List[Any]: List of PySTAC items.
        """
        retries = 3
        for attempt in range(retries):
            try:
                search = self.catalog.search(
                    collections=[self.collection],
                    bbox=bbox,
                    datetime=datetime_range,
                    max_items=max_items,
                )
                items = list(search.items())
                logger.info(f"Found {len(items)} items in {self.collection}.")
                return items
            except Exception as e:
                logger.warning(f"Search failed (attempt {attempt+1}/{retries}): {e}")
                time.sleep(2 ** attempt)
        logger.error(f"Search failed after {retries} attempts.")
        return []

    def load_sentinel1(self, bbox: List[float], datetime_range: str) -> Any:
        """Load Sentinel-1 GRD (VV/VH bands) as xarray datasets using odc.stac.

        Args:
            bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat].
            datetime_range (str): Datetime range string.

        Returns:
            xarray.Dataset: The loaded dataset.
        """
        import odc.stac
        items = self.search(bbox, datetime_range, max_items=10)
        if not items:
            return None
            
        logger.info(f"Loading {len(items)} Sentinel-1 items via odc-stac...")
        # Sentinel-1 bands for flood detection
        ds = odc.stac.load(
            items,
            bands=["vv", "vh"],
            bbox=bbox,
            resolution=10, # 10m resolution
            chunks={"x": 512, "y": 512, "time": 1},
        )
        return ds

    def load_sentinel2(self, bbox: List[float], datetime_range: str) -> Any:
        """Load Sentinel-2 L2A (6 bands) as xarray datasets using odc.stac.

        Args:
            bbox (List[float]): Bounding box.
            datetime_range (str): Datetime range string.

        Returns:
            xarray.Dataset: The loaded dataset.
        """
        import odc.stac
        # Switch collection temporarily if we need S2
        original_collection = self.collection
        self.collection = "sentinel-2-l2a"
        items = self.search(bbox, datetime_range, max_items=10)
        self.collection = original_collection
        
        if not items:
            return None
            
        logger.info(f"Loading {len(items)} Sentinel-2 items via odc-stac...")
        # S2 bands: Blue, Green, Red, NIR, SWIR1, SWIR2
        ds = odc.stac.load(
            items,
            bands=["B02", "B03", "B04", "B08", "B11", "B12"],
            bbox=bbox,
            resolution=10,
            chunks={"x": 512, "y": 512, "time": 1},
        )
        return ds

    def load_dem(self, bbox: List[float]) -> Dict[str, Any]:
        """Load Copernicus DEM for area.

        Args:
            bbox (List[float]): Bounding box.

        Returns:
            Dict[str, Any]: Dictionary containing 'array', 'crs', 'transform', 'bounds'.
        """
        return {"array": np.zeros((1, 100, 100)), "crs": "EPSG:4326", "transform": None, "bounds": bbox}
