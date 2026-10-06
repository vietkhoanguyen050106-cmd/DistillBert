import argparse
 
from transformers import DataCollatorWithPadding, Trainer
 
from Bert.config import TEACHER_DIR, TEACHER_EPOCHS, TEACHER_LR
from Bert.dataset import load_sst2
from Bert.Model_teacher import build_teacher
from Bert.utils import compute_metrics, count_parameters, make_training_args, save_history
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TEACHER_EPOCHS)
    parser.add_argument("--lr", type=float, default=TEACHER_LR)
    args = parser.parse_args()
 
    dataset, tokenizer = load_sst2()
    teacher = build_teacher()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
    trainer = Trainer(
        model=teacher,
        args=make_training_args(TEACHER_DIR, epochs=args.epochs, lr=args.lr),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
 
    trainer.train()
    trainer.save_model(TEACHER_DIR)
    tokenizer.save_pretrained(TEACHER_DIR)
    save_history(trainer, TEACHER_DIR)
 
    metrics = trainer.evaluate()
    print(f"[teacher] accuracy={metrics['eval_accuracy']:.4f} "
          f"val_ce_loss={metrics['eval_loss']:.4f} "
          f"params={count_parameters(teacher) / 1e6:.1f}M lr={args.lr} epochs={args.epochs}")
 
 
if __name__ == "__main__":
    main()
    
# USAGE & RATIONALE
# * Trainer (instead of a hand-written loop): mixed precision, evaluation, checkpoints
#   and best-model selection for free; it can be subclassed to change only the loss.
# * Cross-entropy loss: SST-2 is single-label classification.
# * lr 2e-5, 3 epochs, warmup 10%: the standard recipe for fine-tuning BERT; larger
#   learning rates tend to erase pretrained knowledge ("catastrophic forgetting").
# * The saved teacher is the reference ("upper bound") for every student.