import os
import numpy as np
import torch
from torch import nn
import torchvision
import idx2numpy
import matplotlib.pyplot as plt
from BabyNet import BabyNet



def matplotlib_imshow(img, one_channel=False):
  if one_channel:
    img = img.mean(dim=0)
  img = img / 2 + 0.5
  npimg = img.numpy()
  if one_channel:
    plt.imshow(npimg, cmap="Greys")
  else:
    plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()

if __name__ == "__main__":

  label_file = r'./data/t10k-labels-idx1-ubyte'

  training_set = torchvision.datasets.MNIST(
    root="~/fun/MNISTing/data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
  )

  data_classes = tuple(np.unique(idx2numpy.convert_from_file(label_file))) # might need to convert to tuple?

  training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=4,
    shuffle=True,
    num_workers=2
  )


  # Show image grid + labels
  dataiter = iter(training_loader)
  images, labels = dataiter.next()
  img_grid = torchvision.utils.make_grid(images)
  matplotlib_imshow(img_grid, one_channel=True)
  print(data_classes)
  print(labels)
  
  # Intialize model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = BabyNet().to(device)
  




