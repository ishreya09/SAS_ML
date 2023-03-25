import numpy as np
import rasterio as rio

def load_raster(filepath): # filepath of the raster file to be loaded
    '''load a single band raster'''
    with rio.open(filepath) as file: 
        raster = file.read()        
    return raster



def damage_percentage(path):
    
    VV_img= path+"VV.tif"
    VH_img=path+"VH.tif"
    vv=load_raster(VV_img)
    vh=load_raster(VH_img)

    pdi = np.absolute((vv - vh) / (vv + vh))   
    threshold = 0.5

    # Create binary mask
    mask = np.zeros_like(pdi)
    mask[pdi > threshold] = 1

    # Calculate percentage of damage
    num_damaged_pixels = np.sum(mask)
    total_pixels = mask.size
    damage_percentage = num_damaged_pixels / total_pixels * 100
    return damage_percentage

path ="sen12flood\\sen12floods_s1_source\\sen12floods_s1_source\\sen12floods_s1_source_0_2019_03_20\\"
k=damage_percentage(path)
print(k)