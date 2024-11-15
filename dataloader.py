from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from omegaconf import OmegaConf
class GetDataset(Dataset):
    def __init__(self, id_range, runType = 'train', transform=None):
        self.runType = runType
        self.target_masks = []
        self.input_images = []
        config = OmegaConf.load('./params.yaml')
        path = config.data_loader.path
        if runType == 'train':
          self.input_images = [np.array(Image.open(f'{path}train/i{i}.png'), dtype=np.float32) / 255. for i in range(id_range[0], id_range[1])]
          for i in range(len(self.input_images)):
            mask = np.load(f'{path}train/l{i}.npy').astype(dtype=np.float32)
            label = np.asarray([np.where(mask == i, 1, 0) for i in range(0, 5)])
            self.target_masks.append(label)
        elif runType == 'test':
          self.input_images = [np.array(Image.open(f'{path}test/i{i}.png'), dtype=np.float32) / 255. for i in range(id_range[0], id_range[1])]
          for i in range(id_range[0], id_range[1]):
            mask = np.load(f'{path}test/l{i}.npy').astype(dtype=np.float32)
            label = np.asarray([np.where(mask == i, 1, 0) for i in range(0, 5)])
            self.target_masks.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        image = self.transform(image)
        mask = self.target_masks[idx]
        return [image, mask]
