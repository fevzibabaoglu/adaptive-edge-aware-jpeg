# Adaptive Edge-Aware JPEG Compression

This project presents an advanced image compression system designed to enhance the standard JPEG algorithm by introducing a more perceptually-driven approach. The traditional JPEG standard relies on a fixed-grid of 8x8 pixel blocks, which can lead to conspicuous blocking artifacts at higher compression levels and an inefficient allocation of data. This "one-size-fits-all" method often over-compresses detailed, textured regions while using too much space to represent data on smooth, uniform areas.

To address these shortcomings, this system employs a two-pronged strategy: **dynamic block partitioning** and **adaptive quantization**. The core of this method is its ability to analyze the image content before compression. It uses a sophisticated Canny edge detection pipeline to identify areas of high complexity. Based on this edge map, the image is partitioned using a quadtree data structure into variable-sized blocks. Intricate regions with many edges are captured using smaller, more precise blocks, while larger blocks are used to efficiently represent smooth, less complex areas.

This adaptive partitioning is complemented by dynamic quantization. The quantization strength, which determines the level of compression and detail loss, is intelligently adjusted for each block based on its size and user-defined quality settings. This ensures that visually important details are preserved with high fidelity, while uniform regions are compressed more aggressively without significant perceptual loss. The entire system is accessible through an intuitive Graphical User Interface (GUI), which facilitates easy experimentation with parameters and provides real-time visual and metric-based feedback.

***

## Key Features

-   **Adaptive Block Partitioning:** Moves beyond JPEG's rigid 8x8 grid by using a quadtree to partition the image into variable-sized blocks (from 4x4 to 128x128). This content-aware structure allows the compressor to focus on preserving detail precisely where it is needed most.

-   **Dynamic Quantization:** Implements a smarter, adaptive quantization scheme where the quality level is adjusted for each block. Smaller blocks (in detailed areas) receive finer quantization for higher fidelity, while larger blocks (in smooth areas) are quantized more coarsely, optimizing the trade-off between file size and visual quality.

-   **Modern Color Spaces:** Provides support for a range of advanced color spaces (`YCbCr`, `YCoCg`, `OKLAB`, `ICaCb`, `ICtCp`, `JzAzBz`). These spaces offer better separation of luminance and chrominance or greater perceptual uniformity, leading to more effective and visually pleasing compression.

-   **Interactive GUI:** Features a user-friendly interface built with Tkinter that simplifies the complex compression workflow. It allows for easy batch processing, parameter tuning, and provides an immediate preview of the results.

-   **Advanced Quality Assessment:** Integrates standard metrics (PSNR, SSIM) with modern perceptual ones (MS-SSIM, LPIPS) to give a comprehensive and objective measure of compression performance that aligns more closely with human visual perception.

-   **Custom `.ajpg` File Format:** Utilizes a custom file format that encapsulates not only the compressed image data but also all the necessary metadata—including the chosen color space, quality settings, and the unique quadtree structure—ensuring perfect and reproducible decompression.


***

## Getting Started

This section will guide you through the process of setting up the project environment and running the application on your local machine.


### Prerequisites

**Note:** This project was developed and tested using **Python 3.13.2**. While it may work on other Python 3.x versions, compatibility is not guaranteed.


### Installation

Follow these steps to set up the project and install the necessary dependencies. Using a virtual environment is highly recommended to avoid conflicts with other projects.

1.  **Clone the Repository:**
    Clone the project repository:
    ```bash
    git clone https://github.com/fevzibabaoglu/adaptive-edge-aware-jpeg.git
    cd adaptive-edge-aware-jpeg
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    *   On macOS and Linux:
        ```bash
        python -m venv .venv
        source .venv/bin/activate
        ```
    *   On Windows (Powershell):
        ```bash
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
        ```

3.  **Install Dependencies:**
    This project uses a `requirements.txt` file to manage all necessary libraries. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```


### Usage

Once the installation is complete, you can run the application directly from the root of the project directory.

To launch the Graphical User Interface (GUI), execute the following command in your terminal:

```bash
python src/main.py
```

This will open the window, where you can begin compressing and decompressing images.



#### GUI Overview

The application is split into two main panels:

-   **Control Panel (Left):**
    -   **Batch Processing:** Select one or more images (`.png`, `.jpg`, etc.) or `.ajpg` files to compress or decompress.
    -   **Color Space:** Choose the internal color space for compression.
    -   **Quality Range:** Set the minimum and maximum quality values (1-99). The algorithm will interpolate between these based on block size.
    -   **Block Size Range:** Set the minimum and maximum block sizes (as powers of 2, e.g., 4 to 64).
    -   **Action Buttons:** Click `Compress` or `Decompress` to process the files selected in the batch processing area.

-   **Preview Panel (Right):**
    -   **Select Preview Image:** Load a single image for interactive, real-time feedback.
    -   **Update Preview:** Manually refresh the compression result after changing settings.
    -   **Image Comparison:** The top half shows the original image; the bottom half shows the compressed and decompressed result.
    -   **Metrics Display:** Below the images, key quality metrics (PSNR, SSIM, MS-SSIM, LPIPS) and the compression ratio are displayed.


***

## License

This project is licensed under the terms of the **GNU Lesser General Public License v3.0**.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**; without even the implied warranty of **MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE**. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU General Public License and the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

The full text of the licenses can be found in the root of this project:

*   **[COPYING](./COPYING)** (The GNU General Public License)
*   **[COPYING.LESSER](./COPYING.LESSER)** (The GNU Lesser General Public License)


***

## Author and Research Paper

This project, including all source code and the accompanying research, was developed by **Fevzi Babaoğlu (me)**.

The code in this repository is the official implementation for the concepts and algorithms described in the research paper:

**Adaptive Edge-Aware Image Compression: Enhancing JPEG with Dynamic Block Partitioning and Quantization**

The full paper is available within this repository and provides an in-depth explanation of the methodology, evaluation, and results of this work.

*   **[Read the Full Paper (PDF)](./docs/adaptive_edge_aware_jpeg.pdf)**

**If you use this project or its findings in your own research, please consider citing the paper.**
