# Neural DRS
TensorFlow Implementation of DRS Parsing. For details, please refer Discourse Representation Theory and lines of research for DRS Parsing. Especially, the Parallel Meaning Bank.

## Usage
Install the packages.
 ```
 pip install tensorflow-gpu>1.12 OpenNMT-tf==1.25.3
 ```
## Usage

### Train
```
./train.sh
```

### Extract
```
./extract.sh
```

### Evaluation
```
./test.sh
```

## Reference
* Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation (EMNLP 2018), Liu et al. [[paper]](https://arxiv.org/abs/1809.09078)
* lx865712528's EMNLP2018-JMEE repository [[github]](https://github.com/lx865712528/EMNLP2018-JMEE)
* Kyubyong's bert_ner repository [[github]](https://github.com/Kyubyong/bert_ner)
