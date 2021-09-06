# Data set trick in Pytorch

The name of pictures is not the original name from the dataset.

The pictures are actually renamed after val_ssd.py.

The numbers in val(train).txt are actually the real names of the pictures.



## Run the demo
### Run the live MobilenetV1 SSD demo

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
python run_ssd_live_demo.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt 
```

## TODO

1. Resnet34 Based Model.
2. BatchNorm Fusion.
