import argparse
 
from transformers import DataCollatorWithPadding, Trainer
 
from config import STUDENT_DEFAULTS, student_dir
from dataset import load_sst2
from Model_student import build_student
from utils import compute_metrics, count_parameters, make_training_args, save_history
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="scratch", choices=["scratch", "pretrained"])
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    lr = args.lr or STUDENT_DEFAULTS[args.init]["lr"]
    epochs = args.epochs or STUDENT_DEFAULTS[args.init]["epochs"]
    output_dir = student_dir("baseline", args.init, args.layers)
 
    dataset, tokenizer = load_sst2()
    student = build_student(args.init, args.layers)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
    trainer = Trainer(
        model=student,
        args=make_training_args(output_dir, epochs=epochs, lr=lr),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
 
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_history(trainer, output_dir)
 
    metrics = trainer.evaluate()
    print(f"[student_baseline_{args.init}_{args.layers}L] accuracy={metrics['eval_accuracy']:.4f} "
          f"val_ce_loss={metrics['eval_loss']:.4f} "
          f"params={count_parameters(student) / 1e6:.1f}M lr={lr} epochs={epochs}")
 
 
if __name__ == "__main__":
    main()
 
# USAGE & RATIONALE
# * Why a baseline? It is the control experiment: same architecture, same data, same
#   initial weights, same epochs and lr as the distilled student, but NO teacher. The
#   gap baseline -> distilled is exactly what the teacher contributed.
# * From scratch uses lr 1e-4 and 10 epochs: random weights must learn everything, so
#   they need a larger step size and more passes than fine-tuning (lr 2e-5, 3 epochs).
#   These are sensible starting values, not tuned optima; try 5e-5 / 2e-4 if curves are
#   still rising or unstable.
