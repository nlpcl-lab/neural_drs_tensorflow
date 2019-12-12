# Neural DRS
TensorFlow Implementation of DRS Parsing, inspired by original implementation of Neural DRS repository [[github]](https://github.com/RikVN/Neural_DRS)

For details, please refer Discourse Representation Theory and lines of research for DRS Parsing. Especially, the Parallel Meaning Bank.

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
* Abzianidze, Lasha, et al. "The Parallel Meaning Bank: Towards a Multilingual Corpus of Translations Annotated with Compositional Meaning Representations." EACL 2017 (2017): 242.
* van Noord, Rik, et al. "Exploring neural methods for parsing discourse representation structures." Transactions of the Association for Computational Linguistics 6 (2018): 619-633.
* van Noord, Rik, Antonio Toral, and Johan Bos. "Linguistic information in neural semantic parsing with multiple encoders." Proceedings of the 13th International Conference on Computational Semantics-Short Papers. 2019.
