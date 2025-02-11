import numpy as np
import gdal
import os
from itertools import combinations

"""Ce script charge 10 images NDVI (ou moins si spécifié) à partir d'un répertoire d'entrée, génère toutes les combinaisons possibles
de trois bandes parmi ces images, et crée des images empilées pour chaque combinaison. Il utilise GDAL pour manipuler les fichiers
GeoTIFF et préserver les métadonnées géospatiales, et itertools pour générer les combinaisons. Chaque image empilée est sauvegardée
dans un répertoire de sortie sous forme de fichier GeoTIFF multi-bandes """

def load_image(file_path):
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Impossible d'ouvrir le fichier {file_path}.")
    return dataset, dataset.ReadAsArray()

def stack_bands(bands, output_path, reference_dataset):
    driver = gdal.GetDriverByName("GTiff")
    height, width = bands[0].shape
    dataset = driver.Create(output_path, width, height, len(bands), gdal.GDT_Float32)
    dataset.SetGeoTransform(reference_dataset.GetGeoTransform())
    dataset.SetProjection(reference_dataset.GetProjection())
    for i, band in enumerate(bands):
        dataset.GetRasterBand(i + 1).WriteArray(band)
    dataset.FlushCache()

def generate_combinations(file_paths, output_dir, num_bands=3):
    if len(file_paths) < num_bands:
        raise ValueError(f"Pas assez d'images pour générer des combinaisons de {num_bands} bandes.")
    
    
    file_combinations = list(combinations(file_paths, num_bands))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, combo in enumerate(file_combinations):
        datasets = []
        bands = []

        for file_path in combo:
            dataset, band = load_image(file_path)
            datasets.append(dataset)
            bands.append(band)
    
        stacked = np.dstack(bands)
        
        output_path = os.path.join(output_dir, f"stacked_combination_{idx + 1}.tif")
        stack_bands(bands, output_path, datasets[0])
        
        print(f"Combinaison {idx + 1} sauvegardée : {output_path}")

def main(input_dir, output_dir):
    
    file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".tif")]
    
    if len(file_paths) < 10:
        raise ValueError("Le répertoire d'entrée doit contenir au moins 10 images NDVI.")
    
    file_paths = file_paths[:10]
    
    generate_combinations(file_paths, output_dir)

