from datasets import load_dataset
from transformers import AutoTokenizer
 
from config import MAX_LENGTH, TOKENIZER_NAME
 
 
def load_sst2():
    """
    Returns (dataset, tokenizer).
    Every example has: input_ids, attention_mask, label.
 
    Usage:
        dataset, tokenizer = load_sst2()
        print(dataset["train"][0])
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    raw_dataset = load_dataset("stanfordnlp/sst2")
 
    def tokenize_batch(batch):
        # return_token_type_ids=False: DistilBERT has no token-type embeddings and
        # rejects that input; for single sentences BERT's token types are all zeros
        # anyway, so dropping them changes nothing for the teacher.
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=False,
        )
 
    dataset = raw_dataset.map(
        tokenize_batch, batched=True, remove_columns=["sentence", "idx"]
    )
    return dataset, tokenizer

# USAGE & RATIONALE
# * Trainer (instead of a hand-written loop): mixed precision, evaluation, checkpoints
#   and best-model selection for free; it can be subclassed to change only the loss.
# * Cross-entropy loss: SST-2 is single-label classification.
# * lr 2e-5, 3 epochs, warmup 10%: the standard recipe for fine-tuning BERT; larger
#   learning rates tend to erase pretrained knowledge ("catastrophic forgetting").
# * The saved teacher is the reference ("upper bound") for every student.