from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.aideck_ssd import create_aideck_ssd, create_aideck_ssd_predictor
from vision.ssd.chl_ssd_tiny import create_chl_ssd_tiny, create_chl_ssd_tiny_predictor
from vision.ssd.cnn_ssd import create_cnn_ssd, create_cnn_ssd_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch
import onnx
from onnx import shape_inference
import numpy as np


# if len(sys.argv) < 4:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
#     sys.exit(0)
# net_type = sys.argv[1]
# model_path = sys.argv[2]
# label_path = sys.argv[3]
# if len(sys.argv) >= 5:
#     cap = cv2.VideoCapture(sys.argv[4])  # capture from file
# else:
#     cap = cv2.VideoCapture(0)   # capture from camera
#     cap.set(3, 1920)
#     cap.set(4, 1080)

def convert_onnx(net, net_name="/original_mbv2lite.onnx", model_path="./models", version=9,
                 dummy_input=torch.randn(1, *(1, 160, 120), device='cuda' if torch.cuda.is_available() else 'cpu'),
                 out_names=['scores', 'boxes']):
    net.eval()
    torch.onnx.export(net, dummy_input, model_path + net_name, opset_version=version, verbose=False,
                      output_names=out_names)

net_type = 'cnn_ssd'
model_path = 'models/cnn_ssd-Epoch-130.pth'
label_path = 'models/widerperson-model-labels2.txt'
video_path = '../data_set/live_dataset/3.mp4'
cap = cv2.VideoCapture(video_path)  # capture from file
fps = 10
# 获取窗口大小q
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 调用VideoWrite（）函数
videoWriter = cv2.VideoWriter('test_results/3.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                              fps, size)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=0.25, is_test=True)
elif net_type == 'mb3-large-ssd-lite':
    net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-small-ssd-lite':
    net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
elif net_type == 'aideck-ssd':
    net = create_aideck_ssd(len(class_names), is_test=True)
elif net_type == 'chl-ssd-tiny':
    net = create_chl_ssd_tiny(len(class_names), is_test=True)
elif net_type == 'cnn_ssd':
    net = create_cnn_ssd(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)
# pretrained_dict = torch.load(model_path, map_location='cuda:0')
# model_dict = net.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
net = net.to(DEVICE)
# onnx_name = "/aideck_ssd_full.onnx"
# convert_onnx(net, onnx_name, dummy_input=torch.randn(1, 1, 120, 120).to(DEVICE),
#              out_names=['score', 'box'])
# onnx.save(onnx.shape_inference.infer_shapes(onnx.load("models" + onnx_name)), "models" + onnx_name)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=DEVICE)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'aideck-ssd':
    predictor = create_aideck_ssd_predictor(net, candidate_size=200, device=DEVICE)
elif net_type == 'chl-ssd-tiny':
    predictor = create_chl_ssd_tiny_predictor(net, candidate_size=200, device=DEVICE)
elif net_type == 'cnn_ssd':
    predictor = create_cnn_ssd_predictor(net, candidate_size=200, device=DEVICE)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('test', orig_image)
    # cv2.waitKey(0)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.25)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        # label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        label = f"{labels[i]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (int(box[0]) + 20, int(box[1]) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (255, 0, 255),
                    1)  # thickness
    cv2.imshow('annotated', orig_image)
    videoWriter.write(orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
