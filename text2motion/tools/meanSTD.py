import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Assuming you have a custom dataset and a DataLoader
# Replace YourDataset and your_transform with your actual dataset and transforms
# YourDataset should implement the __len__ and __getitem__ methods
# your_transform should be an instance of torchvision.transforms.Compose

# Example:
# class YourDataset(Dataset):
#     # implementation of your dataset here

# your_transform = transforms.Compose([transforms.ToTensor()])  # Adjust as needed

# dataset = YourDataset(transform=your_transform)
# trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def calculate_mean_std_motion(data_loader):
    mean = 0.0
    std = 0.0
    total_samples = 0

    for motions, _ in data_loader:
        batch_size = motions.size(0)
        motions = motions.view(batch_size, motions.size(1), -1)

        mean += motions.mean(2).sum(0)
        std += motions.std(2).sum(0)
        total_samples += batch_size

    mean /= total_samples
    std /= total_samples

    return mean, std


if __name__ == "main":
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, opt.times)


    # Calculate mean and std
    mean, std = calculate_mean_std_motion(trainloader)

    # Save mean and std to files
    np.save('mean.npy', mean.numpy())
    np.save('std.npy', std.numpy())