# Scattered Light Light-Sheet Microscope
This repository documents data analysis for MSci Physics project. We investigate techniques to increase contrast in processed images form a scattered light light-sheet microscope.

## A very brief theory overview
The scattering of light can be harnessed to improve the contrast in processed images. Structures in typical samples are on the order of one-tenth of the illumination wavelength primarily produce Rayleigh scattering, while structures comparable to the wavelength produce Mie scattering. Both regimes depend on the polarisation of the incident light, however they differ significantly in their angular scattering patterns. Mie scattering is approximately isotropic, meaning some scattered light is detected in directions perpendicular to the illumination regardless of polarisation. Rayleigh scattering, however, is highly polarisation-dependent and its intensity can vary sharply depending on the orientation of the electric field relative to the observation direction.

We are particularly interested in investigating scattering occurring perpendicular to the direction of illumination. Consider the example of Rayleigh scattering illustrated below. Incident light passes through a first polariser aligned parallel to the z-axis, such that the associated dipole oscillation occurs along the z-axis, producing scattered light predominantly in the x–y plane. When detecting light perpendicular to the illumination (parallel to the y-axis) above the sample, a second polariser can used to remove light polarised along a certain direction depending on the second polariser's orientation. For example, if this second polariser is also aligned parallel to the z-axis, Rayleigh scattering is allowed to pass through as the polarisation of light and orientation of second polariser are in the same direction, therefore resulting in a brighter signal. If it is rotated vertically, only light polarised parallel to the z-axis can pass through. Rayleigh scattering is blocked due to the light not being polarised in the same direction as the second polariser. Mie scattering, on the other hand, continues to contribute signal in both cases.
<img width="6391" height="5908" alt="Rayleigh" src="https://github.com/user-attachments/assets/e489f76c-e7ad-4c10-933a-1b73784ca216" />

By subtracting images taken with non-aligned polarisers (where Rayleigh scattering is suppressed and the image is dominated by Mie scattering) from images taken with aligned polarisers (where Rayleigh and Mie scattering are present) enhances contrast. The subtraction effectively removes some of the Mie-scattering background.

## Summarised experimental setup
Laser light passes through a first polariser, which is free to rotate, causing the light to be polarised in one direction. Polarised light then passes through a cylindrical lens which bends the light into a thin sheet used to illuminate a thin slice of the sample (see image below). Rayleigh and Mie scattering occurs within this slice. A second polariser is positioned in the y direction (not shown in the image below) 90&deg; above the sample and filters the polarisation of the scattered light before it reaches the camera. Additional optical components are included to maintain the correct focal lengths, prevent aberration and perform dithering.
<img width="8268" height="5378" alt="Setup" src="https://github.com/user-attachments/assets/508156b0-0300-415d-93ee-57e0ae87869f" />

## Documentation
The code in this repository is not fully modular. Each specimen type we analysed required manual adjustments to subplot layouts, scale bars, titles, and other figure-related elements, which made it impractical to generalise the plotting pipeline. 
- [Pictures](Pictures) - Data input: unprocessed images taken from microscope.
  - [#Masks](Pictures/#Masks) - Data input: masks for pollen (generated from imagej).
  - Other subfolders - Data input: each folder (image stack) contains images at a specific concentration of background, at different angles of polarisation.

### Investigation
Make sure to change file paths to your relevant save location. All relevant unprocessed images are saved in the Pictures file.
- [Averages Bar Chart.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/tree/main/Averages%20Bar%20Chart.py) - Investigation into average pixel value and polarisation at varying background concentrations.
- [Balls.py](Balls.py) - Calculates the amount of background suspension required to produce a given concentration, and the mean free path at the given concentration
- [Fish 10x Objective.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/Fish%2010x%20Objective.py) - Generates processed images from data taken using the microscope.
- [Image Analysis.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/Image%20Analysis.py) - Invetsigations into root mean squared (RMS) brightness at varying polarisations.
- [ImageJ analysis.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/ImageJ%20analysis.py) - Investigations into using ImageJ python package: more effective to produce masks within the ImageJ program instead, as it was easier to visualise the mask for low contrast unprocessed images [^*].
- [SC and SNR.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/SC%20and%20SNR.py) - Invenstigation into speckle contrast (SC) and signal to noise ratio (SNR).
- [Save Images.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/SCSave%20Images.py) - Functions to show an averaged stack of iamges, subtract averaged stacks of iamges (taken at different polarisations), and save the subtracted images.
- [data theta.rtf](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/data%20theta.rtf) - Angle of a motorised rotation stage holding a polariser.
  - [Uden navn 2.txt](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/Uden%20navn%202.txt) - Reformatted [data theta.rtf](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/data%20theta.rtf) for easier importing.
  - [scattering profile.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/scattering%20profile.py) - Calculates the scattering profile using [Uden navn 2.txt](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/Uden%20navn%202.txt).
- [everything.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/everything.py) - Amalgumation of [Save Images.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/SCSave%20Images.py), [SC and SNR.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/SC%20and%20SNR.py) with errors and better formatting. One all encompassing function that averages stacks of images, subtracts the averaged stacks, calculates the SC and SB, and saves the plots.
- [functions.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/functions.py) - Investigation into image subtractions at different concentrations.
- [lianna polarisation.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/lianna%20polarisation.py) - Generates degree of polarisation plots for averaged image stacks at given polarisation orientations.
- [mean free path.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/mean%20free%20path.py) - Visualises Rayleight and Mie scattering for the background suspension.
- [read and plot.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/read%20and%20plot.py) - Investigation into the quantitative improvement of contrast after performing image subtraction.

### Plots for report
Code for generating the plots seen in the report.
- [Final Images](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/tree/main/Final%20Images) - Folder of output images for the report. Note that the individual plots generated from [Final Plots 10x Objective.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/Final%20Plots%2010x%20Objective.py) and [Final Plots 40x Objective.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/Final%20Plots%2040x%20Objective.py) are not saved on this repo.
- [Final Plots 10x Objective.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/Final%20Plots%2010x%20Objective.py) - Generates the processed images from data taken using the 10x objective lens.
- [Final Plots 40x Objective.py](https://github.com/LiannaFung/MSci-Scattered-Light-LSM/blob/main/Final%20Plots%2040x%20Objective.py) - Generates the processed images from data taken using the 40x objective lens.
