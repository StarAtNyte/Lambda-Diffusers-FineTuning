# Finetuning Lambda Diffusers

This notebook finetunes lambda-diffusers on carpet pattern variations data. The system generates variations of input images based on guidance from the CLIP model. It includes components for dataset management, image augmentation, and training a neural network model to refine and generate image variations.

## Features
- Image-based pattern generation using Stable Diffusion and CLIP models.
- Support for generating multiple variations of an image based on input guidance.
- Built-in dataset loading and augmentation using Albumentations.
- Fine-tuning capability for training on custom image pairs.


## Usage

### ImageGuidedPatternGenerator Class

This class is the core of the image-guided pattern generation process. It takes an input image and generates multiple variations based on a pre-trained Stable Diffusion pipeline.

```python
generator = ImageGuidedPatternGenerator()
variations = generator.generate_variations(input_image, num_variations=5)
```

### NestedPatternDataset Class

This class loads image pairs from the dataset. It assumes the images are in pairs where one image represents the input and the other represents a variation.

```python
dataset = NestedPatternDataset(root_dir="path/to/dataset")
```

### FineTuner Class

This class fine-tunes the generator model on custom datasets. You can specify parameters like batch size, learning rate, and the subset of the dataset to train on.

```python
fine_tuner = FineTuner(generator_model=generator, train_config=train_config)
fine_tuner.train_epoch()
```

## Training Configuration

The training configuration is provided via the `train_config` dictionary. Here is an example configuration:

```python
train_config = {
    "dataset_path": "path/to/dataset",
    "batch_size": 4,
    "learning_rate": 1e-5,
    "target_size": (256, 256),
    "subset_fraction": 0.5,
    "num_workers": 2
}
```

## Dataset Format

The dataset consists of image pairs, where the input image is stored with a `0` suffix (e.g., `image_0.jpg`), and the variations are stored with other suffixes (e.g., `image_1.jpg`, `image_2.jpg`, etc.). These pairs are used to train the generator to learn how to generate image variations.

## Results
![image](https://github.com/user-attachments/assets/03fdedbe-52bc-4566-a9bf-0e7233154c99)
