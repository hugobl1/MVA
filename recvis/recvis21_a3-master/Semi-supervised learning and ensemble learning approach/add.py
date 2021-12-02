import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

##On définit la transformation adaptée sur les données non étiquetées
data_transforms_add = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

##On charge les données non étiquetées
add_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('imageunlabeled2/',
                         transform=data_transforms_add),
    batch_size=1, shuffle=False, num_workers=1)





model.eval()

classes = []
names=[]
i=0
for data, target in add_loader:
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    output = F.softmax(model(data))
    outputcpu=output.cpu()
    outputnp=outputcpu.detach().numpy()
    foutput=outputnp[0]
    name=add_loader.dataset.imgs[i][0]
    if(max(foutput)>0.95):
        names=names+[name]
        classes=classes+[np.argmax(foutput)]
    i+=1
    print(i)


npclasses=np.array(classes)
npnames=np.array(names)


#trans = transforms.ToPILImage(mode='RGB')
x=add_loader.dataset[0][0]
#z=trans(z)
plt.imshow(x)
plt.show()


