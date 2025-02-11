import numpy as np
import gdal
import os
from skimage import exposure, segmentation, morphology
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

"""
Ce script traite des images satellitaires du satellite VENµS pour délimiter des parcelles agricoles.
Il effectue plusieurs corrections (atmosphérique, radiométrique, géométrique) pour améliorer la qualité des données,
calcule des indices de végétation (NDVI, SAVI) pour enrichir l'analyse, et applique une segmentation optimisée
(supervisée ou non supervisée) avec des algo basics en gise d'exemple pour verifier la qualiter du traitement. Le résultat est sauvegardé sous forme d'un fichier GeoTIFF.
Le script est conçu pour être modulaire et adaptable à d'autres types de données satellitaires.
"""

def load_venus_image(file_path):
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError("Impossible d'ouvrir le fichier image.")
    bands = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.dstack(bands)

def atmospheric_correction(image, metadata):

    corrected_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        gain = metadata['gain'][i]
        offset = metadata['offset'][i]
        corrected_image[:, :, i] = image[:, :, i] * gain + offset
    return corrected_image

def radiometric_correction(image):
    
    for i in range(image.shape[2]):
        band = image[:, :, i]
        band = (band - np.min(band)) / (np.max(band) - np.min(band))
        image[:, :, i] = band
    return image

def geometric_correction(image, dem):
    
    smoothed_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        smoothed_image[:, :, i] = gaussian_filter(image[:, :, i] * (1 + dem / 1000), sigma=1)
    return smoothed_image

def enhance_contrast(image):
    
    for i in range(image.shape[2]):
        image[:, :, i] = exposure.equalize_adapthist(image[:, :, i], clip_limit=0.03)
    return image

def compute_vegetation_indices(image):
    
    nir = image[:, :, 3]  # Bande proche infrarouge (VENµS)
    red = image[:, :, 2]  # Bande rouge
    ndvi = (nir - red) / (nir + red + 1e-6)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + 1e-6)
    return np.dstack((image, ndvi, savi))

def segment_parcels(image, training_labels=None):
    
    if training_labels is not None:
        
        X = image.reshape(-1, image.shape[2])
        y = training_labels.reshape(-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        
        segmented = clf.predict(X).reshape(image.shape[:2])
    else:
        
        gray = np.mean(image[:, :, :4], axis=2) 
        distance = morphology.local_maxima(gray)
        labels = segmentation.watershed(-gray, markers=distance, mask=gray > 0.1)
        segmented = morphology.label(labels)
    return segmented

def optimize_segmentation(segmented):
    
    cleaned = morphology.remove_small_objects(segmented, min_size=50)
    return morphology.relabel_sequential(cleaned)[0]

def save_segmentation(segmented, output_path):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, segmented.shape[1], segmented.shape[0], 1, gdal.GDT_Int32)
    dataset.GetRasterBand(1).WriteArray(segmented)
    dataset.FlushCache()

def main(image_path, dem_path, output_path, metadata, training_labels_path=None):
    
    image = load_venus_image(image_path)
    
    
    dem = gdal.Open(dem_path, gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
    
    
    image = atmospheric_correction(image, metadata)
    image = radiometric_correction(image)
    image = geometric_correction(image, dem)
    image = enhance_contrast(image)
    
    
    image = compute_vegetation_indices(image)
    
    
    training_labels = None
    if training_labels_path:
        training_labels = gdal.Open(training_labels_path, gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
    
    
    segmented = segment_parcels(image, training_labels)
    
    
    segmented = optimize_segmentation(segmented)
    
    
    save_segmentation(segmented, output_path)


if __name__ == "__main__":
    image_path = "venus_image.tif"
    dem_path = "dem.tif"
    output_path = "segmented_parcels.tif"
    metadata = {
        "gain": [0.95, 0.98, 1.0, 1.05],  
        "offset": [0.1, 0.05, 0.0, -0.1] 
    }
    training_labels_path = "training_labels.tif"  
    main(image_path, dem_path, output_path, metadata, training_labels_path)