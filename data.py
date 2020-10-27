from pathlib import Path

import pytorch_lightning as pl
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_dataset
from tokenizers.processors import BertProcessing
from transformers import RobertaForMaskedLM, RobertaConfig, DataCollatorForLanguageModeling, RobertaTokenizer


class WikiDataModule(pl.LightningDataModule):

    def __init__(
        self,
        tokenizer_name_or_path,
        batch_size=32,
        num_workers=8,
        mlm_probability=.15,
        output_dir=None,
        files=None,
        max_vocab_size=None,
        min_frequency=None,
        special_tokens=None
    ):

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mlm_probability = mlm_probability
        self.output_dir = output_dir
        self.files = files
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens


    def prepare_data(self):
        if not Path(self.tokenizer_name_or_path).exists():
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(
                self.files,
                vocab_size=self.max_vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=self.special_tokens
            )
            Path(self.tokenizer_name_or_path).mkdir(parents=True, exist_ok=True)
            tokenizer.save_model(self.tokenizer_name_or_path)

    def setup(self, stage):
        tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name_or_path)
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:3%]').train_test_split(.1)
        ds = ds.filter(lambda example: example['text'] != '' and not example['text'].startswith(' ='))
        ds = ds.map(lambda ex: tokenizer(ex['text'], padding=True, truncation=True, max_length=56), batched=True)
        ds.set_format('torch', columns=['input_ids'])
        self.ds, self.tokenizer = ds, tokenizer
        self.collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=self.mlm_probability)

    def train_dataloader(self):
        return DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.ds['test'], batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)



if __name__ == '__main__':
    # train_tokenizer()
    tokenizer = RobertaTokenizer.from_pretrained('wiki-test-2')

    dm = WikiDataModule(
        tokenizer_name_or_path='wiki-test-3',
        files = ["wikitext-103-raw/wiki.test.raw"],
        max_vocab_size = 30000,
        min_frequency = 2,
        special_tokens = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]
    )

    dm.prepare_data()
    dm.setup('fit')

