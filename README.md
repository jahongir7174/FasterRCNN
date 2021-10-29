FasterRCNN reimplementation using PyTorch

#### Train
* Configure your dataset paths in `main.py`
* Run `python -m torch.distributed.launch --nproc_per_node $ main.py` for training, `$` is number of GPUs


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