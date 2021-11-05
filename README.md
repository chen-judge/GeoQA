
# GeoQA 

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, Liang Lin. “GeoQA: A Geometric Question Answering Benchmark Towards Multimodal Numerical Reasoning”. Annual Meeting of the Association for Computational Linguistics (Findings of ACL 2021). [<a href="https://arxiv.org/abs/2105.14517">PDF</a>]

Download GeoQA benchmark: [<a href="https://www.dropbox.com/sh/9euqc78dvz6a01e/AAAabmDL1Dd8_2dx1wSJSi8Fa?dl=0">Dropbox</a>]



## Environment
python=3.6

allennlp==0.9.0

Document for <a href="http://docs.allennlp.org/v0.9.0/index.html">allennlp</a>
## Usage of NGS-Auxiliary


### Preparing

git clone https://github.com/chen-judge/GeoQA.git

cd GeoQA

pip install -r requirements.txt

Download the <a href="https://www.dropbox.com/sh/9euqc78dvz6a01e/AAAabmDL1Dd8_2dx1wSJSi8Fa?dl=0">data.zip</a>, move it to GeoQA path, and unzip it.


### Training
    
    allennlp train config/NGS_Aux.json --include-package NGS_Aux -s save/test

### Evaluation
Evaluate your trained model:
    
    allennlp evaluate save/test  data/GeoQA2.2/test.pk --include-package NGS_Aux_test --cuda-device 0

Or, you can use our checkpoint <a href="https://www.dropbox.com/sh/9euqc78dvz6a01e/AAAabmDL1Dd8_2dx1wSJSi8Fa?dl=0">NGS_Aux_CKPT.zip</a>, move it to save path, unzip it, and run:

    allennlp evaluate save/NGS_Aux_CKPT  data/GeoQA2.2/test.pk --include-package NGS_Aux_test --cuda-device 0
    
The result of our checkpoint should be:

| Method | Acc | Angle | Length | Other | 
| --- | --- |  --- | --- |--- |
| NGS-Auxiliary | 60.7 | 72.0 | 47.0 | 44.4 |




