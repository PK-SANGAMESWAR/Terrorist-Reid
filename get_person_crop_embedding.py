import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torchreid.models.osnet import osnet_x1_0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = osnet_x1_0(
    num_classes = 1000,
    pretrained = False,
    loss = 'softmax'
)
state_dict = torch.load('reid_model/osnet_x1_0_msmt17.pth', map_location = device)

state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith('classifier')
}
model.load_state_dict(state_dict, strict = False)
model.to(device)
model.eval()

inf_transforms = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_img_for_inference(path):
    img = Image.open(path).convert('RGB')
    return inf_transforms(img)

def get_osnet_1x_embedding(img):
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    img_t = inf_transforms(img)
    person_crop = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        osnet_1x_embedding = model(person_crop)
    return osnet_1x_embedding