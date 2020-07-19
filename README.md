# Train YOLOv5 on custom data

## Todo
1. Clone this repo
2. Download the dataset [(Link to the Dataset)](https://www.kaggle.com/c/global-wheat-detection/data)
3. Unzip the data into the train_data folder in root directory
4. Create a directory structure like below
```
train_data/
	data/
		|-- train
		|   |-------- images
		|   |-------- labels
		|-- val
		|   |-------- images
		|   |---------labels
```
5. Run `python3 preprocess_data.py`
6. Follow https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data