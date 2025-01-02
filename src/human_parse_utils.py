import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import torch

def generate_human_parse(image_path):
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    parse_map = output.argmax(0).numpy()
    return parse_map
