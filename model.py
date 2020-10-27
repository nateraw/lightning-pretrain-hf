import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig


from transformers import AdamW, get_linear_schedule_with_warmup


class Pretrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        config,
        tokenizer,
        warmup_steps=0,
        learning_rate=1e-4,
        adam_epsilon=1e-8,
        weight_decay=0.0
    ):
        super().__init__()
        self.save_hyperparameters('warmup_steps', 'learning_rate', 'adam_epsilon', 'weight_decay')
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits = self(**batch)
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]
