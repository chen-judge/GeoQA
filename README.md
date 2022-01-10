
# GeoQA 

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, Liang Lin. “GeoQA: A Geometric Question Answering Benchmark Towards Multimodal Numerical Reasoning”. Annual Meeting of the Association for Computational Linguistics (Findings of ACL 2021). [<a href="https://arxiv.org/abs/2105.14517">PDF</a>]

Download GeoQA benchmark: [<a href="https://drive.google.com/drive/folders/1fiLTJUq7EPiZHs6AxundNfNEDLw4gtP5?usp=sharing">Google Drive</a>]

01/10/2022 Update: We find some minor errors in data annotation. These errors have been fixed, and both paper on arXiv and this project have been updated.

## Environment
python=3.6

allennlp==0.9.0

Document for <a href="http://docs.allennlp.org/v0.9.0/index.html">allennlp</a>
## Usage of NGS-Auxiliary


### Preparing

git clone https://github.com/chen-judge/GeoQA.git

cd GeoQA

pip install -r requirements.txt

Download the <a href="https://drive.google.com/drive/folders/1fiLTJUq7EPiZHs6AxundNfNEDLw4gtP5?usp=sharing">data.zip</a>, move it to GeoQA path, and unzip it.


### Training
    
    allennlp train config/NGS_Aux.json --include-package NGS_Aux -s save/test

### Evaluation
Evaluate your trained model:
    
    allennlp evaluate save/test  data/GeoQA3/test.pk --include-package NGS_Aux_test --cuda-device 0

Or, you can use our checkpoint <a href="https://drive.google.com/drive/folders/1fiLTJUq7EPiZHs6AxundNfNEDLw4gtP5?usp=sharing">NGS_Aux_CKPT.zip</a>, move it to save path, unzip it, and run:

    allennlp evaluate save/NGS_Aux_CKPT  data/GeoQA3/test.pk --include-package NGS_Aux_test --cuda-device 0
    
The result of our checkpoint should be:

| Method | Acc | Angle | Length | Other | 
| --- | --- |  --- | --- |--- |
| NGS-Auxiliary | 60.0 | 71.5 | 48.8 | 29.6 |




