import os
import numpy as np
import torch
from torch import nn
import idx2numpy
import matplotlib.pyplot as plt


image_file = r'./data/t10k-images-idx3-ubyte'
label_file = r'./data/t10k-labels-idx1-ubyte'

test_images = idx2numpy.convert_from_file(image_file)
test_labels = idx2numpy.convert_from_file(label_file)
#print(images)



x1 = torch.from_numpy(np.array(test_images[0]))
print(x1)
print(f"Shape of tensor", x1.shape)
print(f"Datatype of tensor", x1.dtype)
print(f"Device of tensor", x1.device)

#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#train_featuers, train_labels, 
#print(test_images)
#print(test_labels)

class BabyNet(nn.Module):
  def __init__(self):
    super(BabyNet, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10),
    )
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits


device = "cuda" if torch.cuda.is_available() else "cpu"
model = BabyNet().to(device)
print(model)

X = torch.rand(1, 28,28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print("X: ", X, len(X))
print("Predicted class: ", y_pred)

'''
i =0
while i <= 5:
  img = test_images[i].squeeze()

  plt.imshow(img, cmap=plt.cm.binary)
  label = test_labels[i]

  print("Label: ", label)


  plt.show()
  i+=1
 '''
