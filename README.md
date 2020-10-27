# lightning-pretrain-hf

Install reqs

```
pip install -r requirements.txt
```

Download wiki dataset

```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

Pretrain Roberta on Wikipedia

```
python -i train.py --gpus 1 --precision 16 --max_epochs 5