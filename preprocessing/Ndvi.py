import numpy as np
import gdal
import os
from multiprocessing import Pool, cpu_count
from functools import partial

"""Ce script calcule le NDVI (Normalized Difference Vegetation Index) à partir d'images satellitaires GeoTIFF,
en optimisant les performances et en gérant les données volumineuses. Il inclut plusieurs étapes clés :
chargement et validation des images, application d'un masque de nuages (optionnel), normalisation des bandes
pour réduire les biais, calcul parallèle du NDVI pour accélérer le traitement, gestion des valeurs aberrantes,
et sauvegarde des résultats avec les métadonnées géospatiales."""


def load_image(file_path):
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError("Impossible d'ouvrir le fichier image.")
    bands = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.dstack(bands)

def validate_bands(image, required_bands=2):
    if image.shape[2] < required_bands:
        raise ValueError(f"L'image doit contenir au moins {required_bands} bandes (rouge et proche infrarouge).")
    return image

def apply_cloud_mask(image, cloud_mask):
    if cloud_mask is not None:
        if cloud_mask.shape != image.shape[:2]:
            raise ValueError("Le masque de nuages doit avoir les mêmes dimensions que l'image.")
        image = np.where(cloud_mask[..., np.newaxis], np.nan, image)
    return image

def normalize_bands(image):
    normalized = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):
        band = image[:, :, i]
        min_val, max_val = np.percentile(band, [2, 98])  # Évite les valeurs extrêmes
        normalized[:, :, i] = (band - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)

def compute_ndvi_chunk(chunk, red_idx, nir_idx):
    red = chunk[:, :, red_idx]
    nir = chunk[:, :, nir_idx]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def parallel_ndvi(image, red_idx, nir_idx, chunk_size=1000):
    height, width = image.shape[:2]
    ndvi = np.zeros((height, width), dtype=np.float32)
    
    
    chunks = [
        image[i:i+chunk_size, j:j+chunk_size, :]
        for i in range(0, height, chunk_size)
        for j in range(0, width, chunk_size)
    ]
    
    
    with Pool(cpu_count()) as pool:
        results = pool.map(partial(compute_ndvi_chunk, red_idx=red_idx, nir_idx=nir_idx), chunks)
    
    
    idx = 0
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            chunk_height = min(chunk_size, height - i)
            chunk_width = min(chunk_size, width - j)
            ndvi[i:i+chunk_height, j:j+chunk_width] = results[idx]
            idx += 1
    
    return ndvi

def handle_outliers(ndvi, lower_bound=-1, upper_bound=1):
    return np.clip(ndvi, lower_bound, upper_bound)

def save_ndvi(ndvi, output_path, reference_dataset):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, ndvi.shape[1], ndvi.shape[0], 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(reference_dataset.GetGeoTransform())
    dataset.SetProjection(reference_dataset.GetProjection())
    dataset.GetRasterBand(1).WriteArray(ndvi)
    dataset.FlushCache()

def compute_ndvi(file_path, output_path, red_idx=2, nir_idx=3, cloud_mask_path=None):
    
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    image = load_image(file_path)
    
   
    image = validate_bands(image, required_bands=max(red_idx, nir_idx) + 1)
    
    
    cloud_mask = None
    if cloud_mask_path and os.path.exists(cloud_mask_path):
        cloud_mask = gdal.Open(cloud_mask_path, gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
    image = apply_cloud_mask(image, cloud_mask)
    
    
    image = normalize_bands(image)
    
    
    ndvi = parallel_ndvi(image, red_idx, nir_idx)
    
    
    ndvi = handle_outliers(ndvi)
    
  
    save_ndvi(ndvi, output_path, dataset)

def main(input_dir, output_dir, red_idx=2, nir_idx=3, cloud_mask_dir=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".tif"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"NDVI_{file_name}")
            
            cloud_mask_path = None
            if cloud_mask_dir:
                cloud_mask_path = os.path.join(cloud_mask_dir, file_name)
                if not os.path.exists(cloud_mask_path):
                    cloud_mask_path = None
            
            print(f"Traitement de {file_name}...")
            compute_ndvi(input_path, output_path, red_idx, nir_idx, cloud_mask_path)
            print(f"NDVI calculé et sauvegardé : {output_path}")

