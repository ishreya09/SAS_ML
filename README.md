# Flood detection using Synthetic Aperture Radar and Optical Images from Sentinel 1 and 2

## About Dataset

SEN12-FLOOD

These last decades, Earth Observation brought quantities of new perspectives from geosciences to human activity monitoring. As more data became available, artificial intelligence techniques led to very successful results for understanding remote sensing data. Moreover, various acquisition techniques such as Synthetic Aperture Radar (SAR) can also be used for problems that could not be tackled only through optical images. This is the case for weather-related disasters such as floods or hurricanes, which are generally associated with large clouds cover. Yet, machine learning on SAR data is still considered challenging due to the lack of available labeled data. This dataset is composed of co-registered optical and SAR images time series for the detection of flood events.

Downloaded from Radiant ML Hub : https://mlhub.earth/data/sen12floods

Citation
Clément Rambour, Nicolas Audebert, Elise Koeniguer, Bertrand Le Saux, Michel Crucianu, Mihai Datcu, September 14, 2020, "SEN12-FLOOD : a SAR and Multispectral Dataset for Flood Detection ", IEEE Dataport, doi: https://dx.doi.org/10.21227/w6xz-s898.

N. Anusha, B. Bharathi,Flood detection and flood mapping using multi-temporal synthetic aperture radar and optical data,The Egyptian Journal of Remote Sensing and Space Science,Volume 23, Issue 2, 2020,Pages 207-219,ISSN 1110-9823, https://doi.org/10.1016/j.ejrs.2019.01.001.

## About the project

Floods are one of the most frequently occurring natural catastrophic events impacting human lives, infrastructure and environment around the globe. There will be a long time impact of flood on the region when heavy rainfall occurs for a very long time or huge water flows from the upstream. In order to determine the flood extent, the data can be collected directly on the field or through remote sensing (aerial or satellite). The flood extent determined through in-situ collection can be inaccurate, impractical and costly. The information collected through aerial imagery can have limited spatial and temporal resolution while being expensive to acquire. Furthermore, the gauge stations measure the water height but not the extent of flood. Whereas, the satellite imagery can determine the extent of flooding over large geographical areas covering inaccessible areas in frequent intervals of time.

Though it is impossible to avoid risks of floods or prevent their occurrence, it is quite possible to reduce their effects and the resultant losses. As soon as the information of a flood event is obtained, the earliest available satellite is programmed to collect the required data for the delineation of flooded areas. Due to their synoptic coverage, satellite based imageries are the best tools to assess the extent of flood affected areas with good spatial resolution and it also enables a permanent recording of such events providing advantages over in-situ and other data sources.

## Data used and methodology

This dataset has been constituted to train a new architecture of neural networks for dual-mode and multi-temporal flood classification. We provide an in-depth study of the various components of the model. Indeed, our goal is to assess the relevance of each modality and the contribution of temporal analysis. First, SAR images are expected to help the ground classification generally conducted on multispectral data. 

For example, the normalized water difference index (Gao, 1996) is widely used to detect the presence of water bodies. However, depending on the sensor, this index may suffer from one drawback: bands associated with the near-infrared and short-waved infrared can present a loss of resolution compared to the RGB ones. On the other hand, SAR images are more sensitive to the geometrical distribution of the backscattering elements. 

For instance, smooth, plane surfaces such as roads or open water areas behave as mirrors and backscatter most of the transmitted wave in the specular direction from the sensor. These surfaces produce typical dark areas in the resulting SAR images, allowing to identify these classes quickly. Moreover, polarization is also affected by the presence of water, and statistical approaches combining the VV and VH bands have shown promising performances (Cazals et al., 2016). Finally, the time consistency is essential to distinguish floods from permanent elements like water bodies. So, multitemporal analysis is the key for the detection of abnormal events such as natural disasters and even their prediction ahead of time. This is becoming more and more necessary to avoid potential harms. 


## What exactly is SAR and why to use?

SAR is the remote sensing system that is not dependent on the sun’s electromagnetic energy or the thermal properties of the earth. Compared to optical data, microwave (SAR) satellites data is the preferred tool for flood mapping from space due to their capability of capturing the images day/night, irrespective of the weather conditions. The SAR systems operate in the micro-wave band, which are long waves and have the capability to penetrate through clouds, to some degree of vegetation, rain showers, fog and snow. Also, SAR’s frequent revisits make it ideal for flood monitoring.

Active sensors transmit a signal and receive the backscatter characteristics of different surface features (Martinis and Rieke, 2015). The strength of the radar backscatter depends on multiple factors, notably surface roughness, dielectric properties, and local topography in relation to the radar look angle. 

The smooth open water surface acts as a specular reflector of the radar pulse, which scatters the radar energy away from the sensor, resulting in minimal signal returned to the satellite. As a result, the stagnant water pixels appear dark in radar data which is in contrast with non-water areas. This makes water pixel differentiation and detection easy in the radar data. 

Due to its good capability to distinguish water with other land surfaces, SAR data is more suitable for flood detection. Sentinel-1 twin satellites are one of the best examples for SAR systems.

## 


