from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import RobertaConfig, RobertaForMaskedLM

from data import WikiDataModule
from model import Pretrainer


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    dm = WikiDataModule(
        tokenizer_name_or_path='wiki-test',
        files = ["wikitext-103-raw/wiki.train.raw"],
        max_vocab_size = 30000,
        min_frequency = 2,
        special_tokens = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
        batch_size=args.batch_size
    )
    dm.prepare_data()
    dm.setup('fit')
    tokenizer = dm.tokenizer
    config = RobertaConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        vocab_size=len(tokenizer.get_vocab()),
    )
    hf_model = RobertaForMaskedLM(config)
    model = Pretrainer(hf_model, config, tokenizer)
    trainer = pl.Trainer.from_argparse_args(args, logger=WandbLogger())
    model.hparams.total_steps = (
        (len(dm.ds['train']) // (args.batch_size * max(1, (trainer.num_gpus or 0))))
        // trainer.accumulate_grad_batches
        * float(trainer.max_epochs)
    )
    trainer.fit(model, dm)
