from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.aideck_ssd import create_aideck_ssd, create_aideck_ssd_predictor
from vision.ssd.cnn_ssd import create_cnn_ssd, create_cnn_ssd_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch
import onnx
from onnx import shape_inference
import numpy as np

def convert_onnx(net, net_name="/original_mbv2lite.onnx", model_path="./models", version=9,
                 dummy_input=torch.randn(1, *(1, 160, 120), device='cuda' if torch.cuda.is_available() else 'cpu'),
                 out_names=['scores', 'boxes']):
    net.eval()
    torch.onnx.export(net, dummy_input, model_path + net_name, opset_version=version, verbose=False,
                      output_names=out_names)

# if len(sys.argv) < 5:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
#     sys.exit(0)
net_type = 'cnn_ssd'
model_path = 'models/cnn_ssd-Epoch-130.pth'
label_path = 'models/widerperson-model-labels2.txt'
image_path = '../data_set/VOCdevkit/VOC_widerperson2/JPEGImages/000264.jpg'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-large-ssd-lite':
    net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-small-ssd-lite':
    net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
elif net_type == 'aideck-ssd':
    net = create_aideck_ssd(len(class_names), is_test=True)
elif net_type == 'cnn_ssd':
    net = create_cnn_ssd(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

net.load(model_path)
net = net.to(DEVICE)
# onnx_name = "/cnn_ssd.onnx"
# convert_onnx(net, onnx_name, dummy_input=torch.randn(1, 1, 128, 160).to(DEVICE),
#              out_names=['score0', 'box0','score1', 'box1','score2', 'box2','score3', 'box3'])
# onnx.save(onnx.shape_inference.infer_shapes(onnx.load("models" + onnx_name)), "models" + onnx_name)


if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'aideck-ssd':
    predictor = create_aideck_ssd_predictor(net, candidate_size=200, device= DEVICE)
elif net_type == 'cnn_ssd':
    predictor = create_cnn_ssd_predictor(net, candidate_size=200, device= DEVICE)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]  # box中存的实际上是左上角坐标pt1和右下角坐标pt2
    cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{labels[i]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (int(box[0]) + 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
