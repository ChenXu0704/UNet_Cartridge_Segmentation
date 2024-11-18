import glob, copy, time, cv2, torch
from omegaconf import OmegaConf
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
from dataloader import GetDataset
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
from UNet import UNET
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
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
  # Claim the training related variables
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
  optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
  # Preparing dataset for training and validation
  train_set = GetDataset(id_range=[0, int(ratio*n_samples)], transform=transform)
  val_set = GetDataset(id_range=[int(n_samples*ratio), n_samples], transform=transform)
  dataloaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle = True, num_workers=0),
               'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)}
  
  #Trainging model
  model = train_model(model, optimizer_ft, dataloaders, exp_lr_scheduler, device, num_epochs=epochs)
  model_name = f"model_batch{batch_size}_step{step_size}_gamma{gamma}_ep{epochs}_lr{lr}.pt"
  
  #Save model
  torch.save(model,f'./model/{model_name}')

#==================================Test part===================================
# Define a function that find the ending points of the firing pin drag
def get_arrow_ends(pred_res):
  arrow_ends = []
  label_id = [4, 1] # colors[4] is for firing pin impression and colors[1] is for firing pin drag

  for i in range(2):
    pos_x = []
    pos_y = []
    label = label_id[i]
    for j in range(pred_res.shape[1]):
      for k in range(pred_res.shape[2]):
        if pred_res[label][j][k] > 2:
          pos_x.append(k)
          pos_y.append(j)

    x_mean = 0 if len(pos_x) == 0 else sum(pos_x) // len(pos_x)
    y_mean = 0 if len(pos_y) == 0 else sum(pos_y) // len(pos_y)
    arrow_ends.append([x_mean, y_mean])
  return arrow_ends

def plot_img_and_mask(input_img, pred_res, mask_img):
  colors = [[0, 0, 0], [173/255, 216/255, 230/255], [139/255, 0, 0], [0, 100/255, 0], [128/255, 0, 128/255]]
  input_img = np.transpose(input_img, (1, 2, 0))
  pred_img = np.zeros((pred_res.shape[1], pred_res.shape[2], 3))
  for i in range(pred_res.shape[0]):
    for j in range(pred_res.shape[1]):
      for k in range(pred_res.shape[2]):
        if pred_res[i][j][k] > 0.5:
          pred_img[j][k] = colors[i]
        # Define the ends of the firing pin drag
  point1, point2 = get_arrow_ends(pred_res)
  point2 = [int(point1[i] + (point2[i] - point1[i]) * 1.5) for i in range(2)]
  # Draw the arrow on the image
  arrow_color = (0, 0, 255)  # BGR color for the arrow (here, blue)
  arrow_thickness = 2
  # Draw a line from point1 to point2
  cv2.line(pred_img, point1, point2, arrow_color, arrow_thickness)

  # Calculate the angle of the arrow
  angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
  # Draw arrowhead
  arrowhead_size = 10
  arrowhead_length = 20

  # Calculate arrowhead points
  arrowhead_point1 = (int(point2[0] - arrowhead_length * np.cos(angle - np.pi/4)),
                            int(point2[1] - arrowhead_length * np.sin(angle - np.pi/4)))

  arrowhead_point2 = (int(point2[0] - arrowhead_length * np.cos(angle + np.pi/4)),
                            int(point2[1] - arrowhead_length * np.sin(angle + np.pi/4)))

  # Draw arrowhead lines
  cv2.line(pred_img, point2, arrowhead_point1, arrow_color, arrow_thickness)
  cv2.line(pred_img, point2, arrowhead_point2, arrow_color, arrow_thickness)
  pred_img = pred_img
  plt.figure(figsize=(12, 12))
  plt.subplot(1, 3, 1, alpha=0.5)
  plt.imshow(input_img.cpu().data.numpy())
  plt.imshow(pred_img, alpha=0.5)
  plt.title('Labelling Overlap with Input')
  plt.subplot(1, 3, 2)
  plt.imshow(mask_img)
  plt.title('Ground Truth')
  plt.subplot(1, 3, 3)
  plt.imshow(pred_img)
  plt.title('Labelling')
  plt.savefig('./data/test/test_img.png')

if __name__ == "__main__":
  config = OmegaConf.load('./params.yaml')
  isTraining = config.training.is_training
  isTesting = config.testing.is_testing
  if isTraining:
    model = training(config)
  if isTesting:
    model_path = config.testing.model_path
    model = torch.load(model_path,map_location=torch.device('cpu') )
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = GetDataset([0,4], 'test',transform = transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    inputs, label = next(iter(test_loader))
    pred = model(inputs)
    pred = pred.data.cpu().numpy()
    test_id = config.testing.test_id
    mask_img = cv2.imread(f'./data/test/m{test_id}.png')
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    plot_img_and_mask(inputs[test_id], pred[test_id], mask_img)
    