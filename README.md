# 2023ADL-HW3

### Dataset and Taiwan LLaMa checkpoint download link
https://drive.google.com/drive/folders/1hyk6DjCQA9lMc0jGrqg7PMtjmrbs_fN5

### Run the Test
```shell
bash ./download.sh
bash ./run.sh ./Taiwan-LLM-7B-v2.0-chat ./adapter_checkpoint ./data/public_test.json ./prediction.json
```

##### When you want to try a single work, see below：

### Train
```shell
bash ./train.sh ./data/train.json
```

### Predict
```shell
bash ./predict.sh ./Taiwan-LLM-7B-v2.0-chat ./adapter_checkpoint ./data/public_test.json ./prediction.json
```