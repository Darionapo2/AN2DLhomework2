from matplotlib import pyplot as plt
import random
import math
import os
import cv2

import numpy as np

import constants


def visualize_random_images(data, n = 5):
    max_images = data.shape[0]
    n = min(n, max_images)

    indices = random.sample(range(max_images), n)
    selected_images = [data[i] for i in indices]

    num_rows = math.ceil(n / 5)

    plt.figure(figsize=(15, num_rows * 2))

    for i, img in enumerate(selected_images):
        plt.subplot(num_rows, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Image {indices[i]}")
    plt.show()


def count_duplicates(images):
    image_hashes = {}

    for i, img in enumerate(images):
        img_hash = hash(img.tobytes())

        if img_hash in image_hashes:
            image_hashes[img_hash].append(i)
        else:
            image_hashes[img_hash] = [i]

    duplicates_count = {
        indices[0]: len(indices)

        for indices in image_hashes.values()
        if len(indices) > 1
    }
    return duplicates_count

def apply_colormap(img):
    colored_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for label, color in constants.color_map.items():
        colored_image[img == label] = color

    return colored_image

def show_duplicates(images):
    duplicates_dict = count_duplicates(images)
    n_duplicated_images = len(duplicates_dict)
    print(f"Found {len(duplicates_dict)} unique images with duplicates.")
    cols = 5
    rows = (n_duplicated_images // cols) + 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
    axes = axes.flatten()

    if len(duplicates_dict) == 1:
        axes = [axes]  # Ensure consistent handling for one row

    i = 0
    for i, (index, count) in enumerate(duplicates_dict.items()):

        img = images[index]

        colored_image = apply_colormap(img)

        axes[i].imshow(colored_image)
        axes[i].set_title(f"Image {index}, {count} duplucates")
        axes[i].axis("off")

    for j in range(i, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def show_images_from_specified_class(labels, selected_class = 4, n = 10, threshold = 0.1):
    selected_images = {}
    indexes = []
    for i, img in enumerate(labels):
        # Count the number of pixels of the selected class
        class_pixels = np.sum(img == selected_class)
        total_pixels = img.size
        class_percentage = class_pixels / total_pixels

        if class_percentage >= threshold:
            selected_images.update({i: img})
            indexes.append(i)

    plt.figure(figsize = (15, 50))
    for k, (i, img) in enumerate(selected_images.items()):

        colored_image = apply_colormap(img)

        plt.subplot(n // 5, 5, k + 1)
        plt.imshow(colored_image)
        plt.title(f'Image {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return indexes

def visualize_images(images, indices):
    num_images = len(indices)
    cols = 5
    rows = (num_images // cols) + 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
    axes = axes.flatten()

    i = 0
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx], cmap = 'gray')
        axes[i].set_title(f'Image {idx}')
        axes[i].axis('off')

    for j in range(i, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_class_distribution(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(8, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Pixel Count')
    plt.title('Class Distribution')
    plt.xticks(classes)
    plt.grid(axis='y')
    plt.show()



def save_dataset_images(data, dirname, img_prefix, colormap=False):
    base_dir = os.getcwd()

    # Create the folder if it doesn't exist
    dataset_dir = os.path.join(base_dir, 'dataset_no_outliers')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Create a subfolder for the specific dataset (train, validation, etc.)
    specific_dir = os.path.join(dataset_dir, dirname)
    if not os.path.exists(specific_dir):
        os.makedirs(specific_dir)

    for i, img in enumerate(data):
        if colormap:
            colored_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            for label, color in constants.color_map.items():
                colored_image[img == label] = color
            img = colored_image

        img_filename = f'{img_prefix}{i}.png'
        img_path = os.path.join(specific_dir, img_filename)
        print(f'Saving image {i}')
        # Save the image using OpenCV
        cv2.imwrite(img_path, img)
