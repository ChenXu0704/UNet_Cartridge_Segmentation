import glob
from omegaconf import OmegaConf
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
import copy 
from dataloader import GetDataset
import time
from torchvision import transforms, datasets, models
import torch 
from torch.utils.data import Dataset, DataLoader
from UNet import UNET
import torch.optim as optim
from torch.optim import lr_scheduler
# The loss function is a combination of BCE and dice loss with given weights.
def calc_loss(pred, target, metrics, bce_weight=0.1):
  target = target.float()
  bce = F.binary_cross_entropy_with_logits(pred, target)

  pred = F.sigmoid(pred)
  dice = dice_loss(pred, target)

  loss = bce * bce_weight + dice * (1 - bce_weight)

  metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
  metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
  metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
  return loss

def print_metrics(metrics, epoch_samples, phase):
  outputs = []
  for k in metrics.keys():
    outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
  print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, dataloaders, scheduler, device, num_epochs=25):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1e10
  train_loss = []
  val_loss = []
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    since = time.time()

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        scheduler.step()
        for param_group in optimizer.param_groups:
          print("LR", param_group['lr'])

        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      metrics = defaultdict(float)
      epoch_samples = 0

      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          loss = calc_loss(outputs, labels, metrics)

          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        epoch_samples += inputs.size(0)

      print_metrics(metrics, epoch_samples, phase)
      epoch_loss = metrics['loss'] / epoch_samples
      if phase == 'train':
        train_loss.append(epoch_loss)
      else:
        val_loss.append(epoch_loss)
      # deep copy the model
      if phase == 'val' and epoch_loss < best_loss:
        print("saving best model")
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

  print('Best val loss: {:4f}'.format(best_loss))
  # load best model weights
  model.load_state_dict(best_model_wts)
  return model

def training(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = UNET(3,5)
  model = model.to(device)
  transform = transforms.Compose([transforms.ToTensor()])
  n_samples = len(glob.glob('./data/train/i*.png'))
  ratio = config.training.train_val_ratio
  batch_size = config.training.batch_size
  step_size = config.training.step_size
  gamma = config.training.gamma
  epochs = config.training.epochs
  lr = config.training.lr
  train_set = GetDataset(id_range=[0, int(ratio*n_samples)], transform=transform)
  val_set = GetDataset(id_range=[int(n_samples*ratio), n_samples], transform=transform)
  dataloaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle = True, num_workers=0),
               'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)}
  optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
  model = train_model(model, optimizer_ft, dataloaders, exp_lr_scheduler, device, num_epochs=epochs)
  model_name = f"model_batch{batch_size}_step{step_size}_gamma{gamma}_ep{epochs}_lr{lr}.pt"
  torch.save(model,f'./model/{model_name}')
if __name__ == "__main__":
  config = OmegaConf.load('./params.yaml')
  isTraining = config.training.is_training
  isTesting = config.testing.is_testing
  if isTraining:
    model = training(config)
  if isTesting:
    pass