import faiss
import numpy as np
import time
import torch
from torchvision import transforms as tsf
from PIL import Image
from model import Bottleneck, ResNet

img_size = 512

transform = tsf.Compose([
    tsf.Resize((img_size, img_size)),
    tsf.ToTensor(),
    tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class EyeDataset(Dataset):
    def __init__(self, img_paths, labels, gray=False, transform=None):
        super(EyeDataset, self).__init__()
        self.img_paths = img_paths
        self.labels = labels
        self.gray = gray
        self.transform = transform
        self.length = len(img_paths)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = Image.open(img_path)
        if self.gray:
            img.convert("L")
        else:
            img.convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

class search:

    def __init__(self, feature, label=None):

        self.feature_db = []
        self.name = []
        self.label=[]
        tstart = time.time()
        self.model = ResNet(Bottleneck, [3, 4, 23, 3], att_type="CBAM")
        self.model.load_state_dict(torch.load(
            "F:\课件\创新实践\model.pth", map_location=torch.device('cpu')))
        self.model.fc1 = torch.nn.Sequential()
        self.model.to(device)
        import csv
        with open(feature) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                self.feature_db.append(row[:-2])
                self.name.append(row[-1])
                self.label.append(row[-2])

        self.cpu_index = faiss.IndexFlatL2(2048)
        # self.cpu_index = faiss.index_cpu_to_all_gpus(
        #     cpu_index)  # make all gpu usable
        # add data(must be float32) to index
        self.feature_db=np.array(self.feature_db, dtype=np.float32)[:,:2048]
        self.cpu_index.add(np.array(self.feature_db, dtype=np.float32))
        elapsed = time.time() - tstart
        print('Completed buliding index in %d seconds' % int(elapsed))

    def search(self, topk, path):
        # img=Image.open(path)
        # img.convert('RGB')
        # img = transform(img)
        # img=[img]
        # img=torch.tensor(img)
        # img = img.to(device, dtype=torch.float32)
        # print(img)
        dataset =EyeDataset([path], torch.tensor([0]), transform=transform)
        loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)
        self.model.eval()
        index=[]
        with torch.no_grad():
            for x,label in loader:
                x = x.to(device, dtype=torch.float32)
                _, pred = self.model(x)
                index=np.array(pred.cpu())
        scores, neighbors = self.cpu_index.search(
            np.array(index).astype('float32'), k=topk)
        result = []
        for i in neighbors[0]:
            result.append("F:\\data\\eye\\train_images\\"+ self.name[i]+".png")

        return result
