# tests/test_dataloader.py

from src.datasets.dataloader import get_data_loader

def test_data_loader():
    data_loader = get_data_loader(
        data_dir='./data/test_dataset',  # Use a small test dataset
        image_size=64,
        batch_size=8,
        num_workers=0
    )
    images, labels = next(iter(data_loader))
    assert images.shape == (8, 3, 64, 64)

