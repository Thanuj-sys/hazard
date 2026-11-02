"""Verify source adapters before reporting real-image classification results."""
import numpy as np
import pytest
import rasterio
import h5py
from scripts.evaluate_real_holdout import load_sample


def write_tif(path, data):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=data.dtype,
                       transform=rasterio.transform.from_origin(10, 20, .01, .01), crs='EPSG:4326') as out:
        out.write(data)


def test_hls_already_reflectance_not_scaled_twice(tmp_path):
    write_tif(tmp_path/'image.tif', np.full((6, 4, 4), .2, dtype='float32'))
    write_tif(tmp_path/'mask.tif', np.ones((4, 4), dtype='int16'))
    image, details = load_sample(tmp_path, {'hazard': 'wildfire', 'files': {'image': 'image.tif', 'mask': 'mask.tif'}})
    np.testing.assert_allclose(image[:6], .2)
    assert not image[6:].any()
    assert details['positive_pixels'] == 16


def test_landslide_undoes_authors_band_mean_normalization(tmp_path):
    means = np.array([1111.81236406,824.63171476,663.41636217,445.17289745,
        645.8582926,1547.73508126,1960.44401001,1941.32229668,
        674.07572865,9.04787384,1113.98338755,519.90397929,20.29228266,772.83144788])
    data = np.broadcast_to(1000/means, (4,4,14)).copy()
    with h5py.File(tmp_path/'image.h5','w') as f: f['img'] = data
    with h5py.File(tmp_path/'mask.h5','w') as f: f['mask'] = np.ones((4,4))
    image, _ = load_sample(tmp_path, {'hazard':'landslide','files':{'image':'image.h5','mask':'mask.h5'}})
    np.testing.assert_allclose(image[:3], .1, rtol=1e-5)
    assert not image[3:].any()


def test_empty_mask_not_assigned_a_positive_hazard_label(tmp_path):
    write_tif(tmp_path/'image.tif', np.full((6,4,4), .2, dtype='float32'))
    write_tif(tmp_path/'mask.tif', np.zeros((4,4), dtype='int16'))
    with pytest.raises(ValueError, match='No positive annotated pixels'):
        load_sample(tmp_path, {'hazard':'wildfire','files':{'image':'image.tif','mask':'mask.tif'}})
