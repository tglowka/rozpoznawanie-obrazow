# Program overview

## Setup

Recommended python version: 3.8

```
# install requirements
pip install -r requirements.txt
```

## Run realtime detection app
Run this command to start capturing camera frames and predicts faces age based on the saved models in the `models` directory. 
```
python -m app.main -c "models"
```
To quit the app press `q`.

## Train model

### Prepare data
First download dataset from: 
https://www.kaggle.com/datasets/jangedoo/utkface-new
Preapre train and test data so that the `data` directory has the following structure inside
```
data
├── face_age_dataset
│   ├── test
│   ├── train
│── face_age_dataset_debug
│   │── test
│   │── train
│
```
The `test` and `train` directories consist of files in format `{FILE_NO}_age_{LABEL}.pt`. File in this format can be generated based on the generation scripts from the existing solution. To generate train and test data please refer to the existing solution report. 

### Run training
There are 2 architectures to choose from: `SIMPLE` (SimpleConvNet 224x224) and `EFF` (EfficientNet 224x224). 
```
python -m train.main --debug -n "EFF" -f 2 -e 2 -l "0.001"
```
To get more information regarding arguments run:
```
python -m train.main -h
```

## Test model
To test model run the command:
```
python -m test.main -c "log/checkpoints" --debug -n "EFF"
```
To get more information regarding arguments run:
```
python -m test.main -h
```