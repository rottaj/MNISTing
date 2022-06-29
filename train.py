import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import idx2numpy
import torchvision
from datetime import datetime
from BabyNet import BabyNet

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

if __name__ == "__main__":
  label_file = r'./data/t10k-labels-idx1-ubyte'

  training_set = torchvision.datasets.MNIST(
    root="~/fun/MNISTing/data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
  )

  validation_set = torchvision.datasets.FashionMNIST('./data/raw', 
    train=False, 
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), 
    download=True
  )

  data_classes = tuple(np.unique(idx2numpy.convert_from_file(label_file))) # might need to convert to tuple?

  training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=4,
    shuffle=True,
    num_workers=2
  )

  validation_loader = torch.utils.data.DataLoader(
    validation_set, 
    batch_size=4, 
    shuffle=False, 
    num_workers=2
  )


  # Show image grid + labels
  dataiter = iter(training_loader)
  images, labels = dataiter.next()

  print(data_classes)
  print(labels)

  model = BabyNet() ### UPDATE THIS

  loss_fn = torch.nn.CrossEntropyLoss()

  # NB: Loss functions expect data in batches, so we're creating batches of 4
  # Represents the model's confidence in each of the 10 classes for a given input
  dummy_outputs = torch.rand(4, 10)
  # Represents the correct class among the 10 being tested
  dummy_labels = torch.tensor([1, 5, 3, 7])

  print(dummy_outputs)
  print(dummy_labels)

  loss = loss_fn(dummy_outputs, dummy_labels)
  print('Total loss for this batch: {}'.format(loss.item()))
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




  # Initializing in a separate cell so we can easily add more epochs to the same run
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
  epoch_number = 0

  EPOCHS = 5

  best_vloss = 1_000_000.

  for epoch in range(EPOCHS):
      print('EPOCH {}:'.format(epoch_number + 1))

      # Make sure gradient tracking is on, and do a pass over the data
      model.train(True)
      avg_loss = train_one_epoch(epoch_number, writer)

      # We don't need gradients on to do reporting
      model.train(False)

      running_vloss = 0.0
      for i, vdata in enumerate(validation_loader):
          vinputs, vlabels = vdata
          voutputs = model(vinputs)
          vloss = loss_fn(voutputs, vlabels)
          running_vloss += vloss

      avg_vloss = running_vloss / (i + 1)
      print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

      # Log the running loss averaged per batch
      # for both training and validation
      writer.add_scalars('Training vs. Validation Loss',
                      { 'Training' : avg_loss, 'Validation' : avg_vloss },
                      epoch_number + 1)
      writer.flush()

      # Track best performance, and save the model's state
      if avg_vloss < best_vloss:
          best_vloss = avg_vloss
          model_path = 'model_{}_{}'.format(timestamp, epoch_number)
          torch.save(model.state_dict(), model_path)

      epoch_number += 1

      
