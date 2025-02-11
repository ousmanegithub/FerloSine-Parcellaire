
import numpy as np
import gdal
import os
import matplotlib.pyplot as plt

"""
Ce script extrait les profils temporels et spectraux à partir d'images NDVI empilées (générées précédemment) et les visualise
sous forme de deux graphiques distincts. Le profil temporel montre l'évolution du NDVI pour chaque bande au fil du temps,
tandis que le profil spectral illustre la variation du NDVI à travers les bandes pour un instant donné. 
"""


def load_stacked_image(file_path):
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Impossible d'ouvrir le fichier {file_path}.")
    return dataset.ReadAsArray()

def extract_temporal_profile(stacked_images, pixel_x, pixel_y):
    temporal_profiles = []
    for image in stacked_images:
        
        profile = [image[i, pixel_y, pixel_x] for i in range(image.shape[0])]
        temporal_profiles.append(profile)
    return np.array(temporal_profiles)

def extract_spectral_profile(stacked_image, pixel_x, pixel_y):
    
    return [stacked_image[i, pixel_y, pixel_x] for i in range(stacked_image.shape[0])]

def plot_temporal_profile(temporal_profiles, output_path):
    plt.figure(figsize=(10, 6))
    for i in range(temporal_profiles.shape[1]):
        plt.plot(temporal_profiles[:, i], label=f'Bande {i+1}', marker='o')
    plt.title("Profil Temporel des NDVI")
    plt.xlabel("Temps (index des images)")
    plt.ylabel("NDVI")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_spectral_profile(spectral_profiles, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(spectral_profiles) + 1), spectral_profiles, marker='o', color='green')
    plt.title("Profil Spectral des NDVI")
    plt.xlabel("Bande")
    plt.ylabel("NDVI")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main(stacked_dir, pixel_x, pixel_y, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    stacked_files = [os.path.join(stacked_dir, f) for f in os.listdir(stacked_dir) if f.endswith(".tif")]
    if not stacked_files:
        raise ValueError("Aucune image empilée trouvée dans le répertoire spécifié.")
    
    stacked_images = [load_stacked_image(f) for f in stacked_files]
    
    
    temporal_profiles = extract_temporal_profile(stacked_images, pixel_x, pixel_y)
    
    
    spectral_profile = extract_spectral_profile(stacked_images[0], pixel_x, pixel_y)
    
    
    temporal_plot_path = os.path.join(output_dir, "temporal_profile.png")
    spectral_plot_path = os.path.join(output_dir, "spectral_profile.png")
    
    plot_temporal_profile(temporal_profiles, temporal_plot_path)
    plot_spectral_profile(spectral_profile, spectral_plot_path)
    
    print(f"Profil temporel sauvegardé : {temporal_plot_path}")
    print(f"Profil spectral sauvegardé : {spectral_plot_path}")

