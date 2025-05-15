import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
from PIL import Image
from get_model import get_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', type=str, default='resnet50', help='cnn')
parser.add_argument('--img_path', type=str, default='results/')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--root_dir', type=str, default='results', help='root path')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# surrogate_model = ['DeiT', 'Inception', 'MNV2', 'ResNet50']
# all_surrogate_model_path = []
# for m in surrogate_model:
#     sm_path = os.path.join(args.root_dir, m)
#     all_surrogate_model_path.append(sm_path) # 获得所有代理模型下不同算法的结果的路径的parent path

# results_paths_DeiT = os.listdir(all_surrogate_model_path[0])
# results_paths_Inception = os.listdir(all_surrogate_model_path[1])
# results_paths_MNV2 = os.listdir(all_surrogate_model_path[2])
# results_paths_ResNet50 = os.listdir(all_surrogate_model_path[3])

# all_algorithms = ['ADEF','cAdv','colorfool','NCF','ReColorAdv','SAE']
# results/DeiT/ADEF-Attack-DeiT

if __name__ == '__main__':
    img_list = os.listdir(args.img_path)
    img_list.sort()
    print('Total Images', len(img_list))
    model_names = ['MobileNetV2','MobileNetV3','ResNet50','ResNet101','EfficientNet','DeiT','simple-vit','Inception','densenet']
    labels_f = open('labels.txt').readlines()
    for model in model_names:
        print(f'Target_model is: {model}')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_size=(224, 224)
        mean = torch.Tensor(mean).cuda()
        std = torch.Tensor(std).cuda()
        
        # Model
        net = get_model(model)
        if device == 'cuda':
            net.to(device)
        net.eval()
        if model == 'Inception':
            image_size = (299, 299)
        cnt = 0
        acc = 0
        for (i, img_p) in enumerate(img_list):
            pil_image = Image.open(os.path.join(args.img_path, img_p)).convert('RGB').resize(image_size)
            img = (torch.tensor(np.array(pil_image), device=device).unsqueeze(0)/255.).permute(0, 3, 1, 2)
            img = img - mean[None,:,None,None]
            img = img / std[None,:,None,None]
            out = net(img.cuda())
            _, predicted = out.max(1)
            idx = int(img_p.split('.')[0]) -1
            label = int(labels_f[idx]) -1
            # print("label:", label, "| predicted",predicted.item())
            if predicted[0] == label:
                acc += 1
        print('-' * 60)
        ASR = 100 - acc / len(img_list) * 100
        print(model, "ASR:", ASR)
