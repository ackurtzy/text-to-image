# Text-to-Image Diffusion Model using Quick, Draw! Dataset 

## Overview 
This repository implements a **Text-to-Image Diffusion Model**  trained on rasterized versions of the [Google Quick, Draw! Dataset]() . The model uses text embeddings to condition a U-Net architecture, which generates images corresponding to input text prompts.
The workflow includes:
 
1. **Downloading and Processing** : Convert vector sketches in NDJSON format into rasterized images (`.npy` files).
 
2. **Training** : A diffusion model conditioned on text embeddings derived from phrases.
 
3. **Inference** : Generate rasterized images conditioned on textual prompts.
 
4. **Visualization** : Plot and animate the diffusion process.


---


## Key Features 

- Processes Quick, Draw! vector images into 28x28 rasterized images.
 
- Text conditioning using **BERT embeddings** .
 
- Implements a **U-Net-based diffusion model**  for generating images.

- Supports model training, inference, and visualization of generated results.


---


## File Structure 


```graphql
text-to-image/
│
├── download_and_convert.py   # Downloads and converts Quick, Draw! data
├── vector_to_raster_lib.py   # Converts vector sketches to raster images
├── train_text.py             # Trains the U-Net diffusion model
├── inference.py              # Generates images conditioned on text
├── plot_data.py              # Visualizes random sketches
├── model_text.py             # U-Net architecture with text embeddings
├── helpers.py                # Helper functions for sampling, diffusion, etc.
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```


---


## Dependencies 

The project requires the following libraries:

- Python 3.8+

- PyTorch

- NumPy

- Transformers (for BERT text embeddings)

- CairoCFFI (for vector-to-raster processing)

- Matplotlib

- TQDM
 
- Google Cloud SDK (`gsutil` for data download)

- einops

### Install Dependencies 

To install all required packages, run:


```bash
pip install -r requirements.txt
```
For `gsutil`, install the [Google Cloud SDK]() .

---


## Setup and Usage 
1. **Downloading and Converting Quick, Draw! Data** The script `download_and_convert.py` automates downloading Quick, Draw! NDJSON files and converts them into rasterized images.
Run:


```bash
python download_and_convert.py
```

- Downloads NDJSON files from Google Cloud Storage.

- Converts vector images to rasterized images (28x28).
 
- Saves them as `.npy` files under `guessed_quickdraw_data`.

> **Note** : Ensure `gsutil` is configured and authenticated.

---

2. **Training the Diffusion Model** 
To train the text-to-image model:


```bash
python train_text.py
```
Configuration (inside `train_text.py`): 
- `data_paths`: Directory containing `.npy` files.
 
- `alternate_embeddings_path`: JSON file with alternate text phrases.
 
- Training parameters like `batch_size`, `learning_rate`, `epochs`, and `diffusion_timesteps`.
The model is saved every 0.5 epochs to the `models/` directory.

---

3. **Inference: Generating Images** Run inference using `inference.py`:

```bash
python inference.py
```

#### Configuration: 
 
- `model_path`: Path to the trained model.
 
- `image_label_paths`: Path to a JSON list of trained categories.
 
- `random_sample`: Set to `True` for random categories, or define specific phrases.
 
- `gif_save_path`: Path to save an animated GIF of the diffusion process.

The script:

- Plots generated images for 9 prompts.

- Saves a GIF showing the image creation process through diffusion.


---

4. **Visualizing Rasterized Data** To plot samples from a processed `.npy` file:

```bash
python plot_data.py
```
Update `data_path` in the script to point to the `.npy` file you want to visualize.

---


## Example Workflow 
 
1. Download and preprocess Quick, Draw! data:


```bash
python download_and_convert.py
```
 
2. Train the model:


```bash
python train_text.py
```
 
3. Generate images:


```bash
python inference.py
```


---


## Results 

- Input: Text prompts like "submarine" or "plane."

- Output: Generated 28x28 grayscale images that match the text prompt.

- Visualization: Plot of generated images and an animated GIF showing the diffusion process.


---


## Acknowledgments 
 
- **Google Quick, Draw! Dataset**  for vector sketches.
 
- **Diffusion Models**  based on concepts from the paper *"Denoising Diffusion Probabilistic Models"*.
 
- **BERT**  embeddings for text conditioning.
