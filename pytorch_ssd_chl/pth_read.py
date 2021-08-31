import torch

path = 'train_results/VOC_widerperson5/20210726-121300-mb2-lite/models/mb2-ssd-lite-Epoch-100-Loss-3.6957368180155754.pth'
pretrained_dict = torch.load(path, map_location='cuda:0')
dict_key = []
for k, v in pretrained_dict.items():  # k 参数名 v 对应参数值
    dict_key.append(k)
# print(dict_key)
W_dict = dict(W_bits=15)
net_dict = dict()
print(W_dict)
i = 0
while dict_key[i].split('.')[0] == 'base_net':
    if dict_key[i].split('.')[-1] == 'weight' and dict_key[i+2].split('.')[-1] != 'running_mean':
        net_dict[dict_key[i].split('.weight')[0]] = W_dict
    i = i+1
print(net_dict)

