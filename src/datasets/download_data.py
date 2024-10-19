# scripts/download_data.py

import os
import opendatasets as od

def download_dataset(dataset_url, data_dir):
    """
    Downloads the dataset from the given URL to the specified directory.

    Args:
        dataset_url (str): The URL of the dataset to download.
        data_dir (str): The directory to save the dataset.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    od.download(dataset_url, data_dir)

if __name__ == '__main__':
    # Example dataset URL (replace with your dataset's URL)
    dataset_url = 'https://www.kaggle.com/datasets/splcher/animefacedataset'
    data_dir = './data'
    download_dataset(dataset_url, data_dir)

