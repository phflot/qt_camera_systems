# QT Camera Systems


![Fig1](img/fig1.jpg)

## Downloads

Clone the repository via
```
$ git clone ssh://git@erlgit.htwsaar.de:2222/snnu/computer-vision/qt_camera_systems.git
```

## Requirements

- **Python 3.8+**  
- **Spinnaker 4.0 SDK** and **Spinnaker 3.8 Python bindings** (for FLIR/PointGrey Thermal Cameras)  
- **XIMEA API** (for XIMEA industrial cameras)  
- **neurovc**  
  - Install via:
    ```
    pip install neurovc
    ```
  This library contains utility functions, in particular for motion magnification (`MagnificationTask`, `AlphaLooper`, ...) and calibration / video IO helpers.

## Contents of this Repository

The project website for the original setup used to develop the calibration can be found [here](https://www.snnu.uni-saarland.de/covid19/). We use a 4x13 circular calibration board with 1.5cm circle diameter which is printed and glued 
onto a metal plate with the same pattern cutout.

- **Calibration and alignment** for multiple cameras (visible, NIR, thermal).  
  - Scripts and classes that handle capturing frames from Ximea and FLIR/PointGrey thermal cameras, performing stereo alignment, and merging landmarks (in `MultiModalMappingWorker`, etc.).  
- **Motion magnification** with a Lagrangian approach, referencing the `neurovc.momag.flow_processing` module.  
- **Threaded architecture** leveraging `PyQt6`:
  - `FrameGrabber`, `ThermalGrabber` classes for camera capture in dedicated QThreads.
  - Separate QThreads for data I/O, calibration, motion magnification, and real-time face or skin segmentation.  
- **Heart rate / rPPG** extraction logic (`HeartRateWorker`) employing a “POS” approach.  
- **Thermal data** mapping and region-of-interest temperature measurement (e.g., eye, mouth).  
- **Calibration pattern**: See `calibration_pattern.pdf` for the 4×13 circular board layout.  
- **GUI Applications**:
  - `VIScreener` for multimodal display of multiple cameras (thermal, NIR, etc.).
  - `Momag` for motion magnification demonstrations.
  - `MomagWebcam` for motion magnification with a webcam.

## Installation

1. **Set up a Conda environment**:
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven’t already, and create a new environment with Python 3.8:
  ```
  conda create -n multimodal python=3.8 -y conda activate multimodal
  ```

2. **Install dependencies**:
- Install requirements via:
  ```
  pip install -r requirements.txt
  ```

3. **Install `Spinnaker` SDK and Python bindings**:
- Install the `Spinnaker SDK` from the resources folder in the repository:
  - [Spinnaker SDK Installer](resources/SpinnakerSDK_FULL_4.0.0.116_x64.exe)
- Install the Python bindings:
    ```
    pip install .\resources\spinnaker_python-4.0.0.116-cp38-cp38-win_amd64\spinnaker_python-4.0.0.116-cp38-cp38-win_amd64.whl
    ```

4. **Install `XIMEA` API**:
- Install the `XIMEA API` from the repository:
  - [XIMEA API Installer](resources/XIMEA_APIInstaller.exe)
- After installation, manually copy the `ximea` Python bindings to the appropriate site-packages directory of the Python environment. The bindings are typically located in:
  ```
  C:\Program Files\XIMEA\API\python\ximea
  ```
- Copy the `ximea` python folder to your Conda environment:
  ```
  <conda-env-path>\Lib\site-packages\
  ```


## Citation

If you use this code in work for publications, please cite in the following way.

**1. Camera routines**:
  
  > Flotho, P., Bhamborae, M., Grun, T., Trenado, C., Thinnes, D., Limbach, D., & Strauss, D. J. (2021). Multimodal Data Acquisition at SARS-CoV-2 Drive Through Screening Centers: Setup Description and Experiences in Saarland, Germany. J Biophotonics.
  
  BibTeX entry
  ```bibtex
  @article{flotea2021b,
      author = {Flotho, P. and Bhamborae, M.J. and Grün, T. and Trenado, C. and Thinnes, D. and Limbach, D. and Strauss, D. J.},
      title = {Multimodal Data Acquisition at SARS-CoV-2 Drive Through Screening Centers: Setup Description and Experiences in Saarland, Germany},
      year = {2021},
    journal = {J Biophotonics},
    pages = {e202000512},
    doi = {https://doi.org/10.1002/jbio.202000512}
  }
  ```

**2. Motion magnification**:

  > Flotho, P., Heiss, C., Steidl, G., & Strauss, D. J. (2023). Lagrangian motion magnification with double sparse optical flow decomposition. Frontiers in Applied Mathematics and Statistics, 9, 1164491.
  
  ```bibtex
  @article{flotho2023lagrangian,
    title={Lagrangian motion magnification with double sparse optical flow decomposition},
    author={Flotho, Philipp and Heiss, Cosmas and Steidl, Gabriele and Strauss, Daniel J},
    journal={Frontiers in Applied Mathematics and Statistics},
    volume={9},
    pages={1164491},
    year={2023},
    publisher={Frontiers Media SA}
  }
  ```

  and for facial landmark-based decomposition:
  
  > Flotho, P., Heiß, C., Steidl, G., & Strauss, D. J. (2022, July). Lagrangian motion magnification with landmark-prior and sparse PCA for facial microexpressions and micromovements. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 2215-2218). IEEE.
  
  ```bibtex
  @inproceedings{flotho2022lagrangian,
    title={Lagrangian motion magnification with landmark-prior and sparse PCA for facial microexpressions and micromovements},
    author={Flotho, Philipp and Hei{\ss}, Cosmas and Steidl, Gabriele and Strauss, Daniel J},
    booktitle={2022 44th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
    pages={2215--2218},
    year={2022},
    organization={IEEE}
  }
  ```

**3. Thermal landmarks:**

  > Flotho, P., Piening, M., Kukleva, A., & Steidl, G. (2024). T-FAKE: Synthesizing Thermal Images for Facial Landmarking. arXiv preprint arXiv:2408.15127.
  
  ```bibtex
  @article{flotho2024t,
    title={T-FAKE: Synthesizing Thermal Images for Facial Landmarking},
    author={Flotho, Philipp and Piening, Moritz and Kukleva, Anna and Steidl, Gabriele},
    journal={arXiv preprint arXiv:2408.15127},
    year={2024}
  }
  ```