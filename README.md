# HGEMD
Heterogeneous Graph Embedding Malware Detection.

for train model, run: 
``` 
python3 train.py --view multi
```

for adversarial attack, run: 
``` 
python3 attack.py jsma 50 --view multi
```

The repository is organised as follows:

- `train.py`  entry script for training model.
- `attack.py`  entry script for adversarial attack.
- `setting.py`  config file.
- `layers/` define layers used in models
- `model_zoo/` define models
- `data/` data path  
- `adv/`  adversarial attacks. jsma and fgsm
