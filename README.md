FasterRCNN reimplementation using PyTorch

#### Train
* Configure your dataset paths in `main.py`
* Run `bash main.sh $ --train` for training, `$` is number of GPUs


#### Dataset structure
    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt
        

#### Reference
* https://github.com/ultralytics/yolov5