# ds6050-group1-project

# Data

## Caltech Pedestrian Dataset
https://data.caltech.edu/records/f6rph-90m20

### Download Data
```bash
wget "https://data.caltech.edu/records/f6rph-90m20/files/data_and_labels.zip?download=1" -O caltechpedestriandataset.zip
```

### Unpack data
```bash
unzip caltechpedestriandataset.zip -d caltechpedestriandataset
```

### Convert videos to images with YOLO annotation format

**Install requirements in base conda environment**
```bash
pip install opencv-python
```

**Run caltech-preprocessing-yolo.py**
```bash
python caltech-preprocessing-yolo.py
```

**Create `/datasets/caltechpedestriandataset.yaml`**
```yaml
path: /full/path/to/datasets/caltechpedestriandataset/
train: /full/path/to/datasets/caltechpedestriandataset/images/train
val: /full/path/to/datasets/caltechpedestriandataset/images/val
    
nc: 1
    
names: [
    'person'
]
```

### Convert to COCO label format

**Install requirements in base conda environment**
```bash
pip install pylabel
```

**Train**
```bash
python yolo-to-coco.py \
    --path_to_annotations /full/path/to/dataset/labels/ \
    --path_to_images /full/path/to/dataset/labels/images/ \
    --path_to_yolo_yaml /full/path/to/dataset.yaml \
    --name caltechpedestriandataset_train
```

### Clean-up

```bash
rm -rf caltechpedestriandataset
```

## EuroCity Persons Dataset
https://eurocity-dataset.tudelft.nl/eval/overview/statistics

### Download Data
See https://eurocity-dataset.tudelft.nl/eval/user/login?_next=/eval/downloads/detection (requires username and password)

### Unpack data
```bash
unzip ECP_<data>.zip
```

### Convert videos to images with YOLO annotation format
TODO

### Convert to COCO label format
TODO

### Clean-up

```bash
rm -rf ECP_<data>.zip
```

# Model

## YOLOv8

**Install requirements in base conda environment**
```bash
pip install ultralytics
```

**Train Model**
```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="/full/path/to/datasets/mydataset.yaml", 
            epochs=epochs, verbose=True, batch=64)
```
