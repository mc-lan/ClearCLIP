from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from clearclip_segmentor import ClearCLIPSegmentation

img = Image.open('images/horses.jpg')
name_list = ['sky', 'hill', 'tree', 'horse', 'grass']

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = ClearCLIPSegmentation(
    clip_type='CLIP',
    vit_type='ViT-B/16',
    model_type='ClearCLIP',
    name_path='./configs/my_name.txt',
    ignore_residual=True,
    prob_thd=0.0,  # need to adjust if background is given
)

seg_pred = model.predict(img_tensor, data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].axis('off')
ax[1].imshow(seg_pred, cmap='viridis')
ax[1].axis('off')
plt.tight_layout()
# plt.show()
plt.savefig('images/seg_pred.png', bbox_inches='tight')