
# GeoQA 

简单介绍下benchmark
以及下载路径


## Install dependencies
```
pip install allennlp==0.9.0 torch==1.2.0 opencv-python
```
## Usage

Run the code of NGS-Auxiliary.

### Preparing

git clone https://github.com/chen-judge/GeoQA.git

cd GeoQA

Download the data.zip, move it to GeoQA path, and unzip it.


### Training
    
    allennlp train config/NGS_Aux.json --include-package NGS_Aux -s save/test

### Evaluation
    
    allennlp evaluate save/test  data/GeoQA2.2/test.pk --include-package NGS_Aux_test --cuda-device 0



