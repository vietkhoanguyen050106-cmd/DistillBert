import argparse
 
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
 
from config import ALPHA, BETA, STUDENT_DEFAULTS, TEACHER_DIR, TEMPERATURE, student_dir
from dataset import load_sst2
from Model_student import build_student
from Model_teacher import build_teacher
from utils import compute_metrics, count_parameters, make_training_args, save_history
 
 
class DistillationTrainer(Trainer):
    """A normal Hugging Face Trainer whose only change is how the loss is computed."""
 
    def __init__(self, *args, teacher, temperature, alpha, beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.to(self.args.device)
        self.teacher.eval()                        # no dropout -> stable targets
        for p in self.teacher.parameters():
            p.requires_grad = False                # frozen: never updated
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        # running sums so the log can show the CE and KD parts separately
        self._ce_sum = torch.zeros((), device=self.args.device)
        self._kd_sum = torch.zeros((), device=self.args.device)
        self._steps = 0
 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        use_hidden = self.beta > 0 and model.training
 
        # (a) Student forward. `labels` is in inputs, so .loss is the CE loss.
        student_out = model(**inputs, output_hidden_states=use_hidden)
        ce_loss = student_out.loss.mean()
 
        # Evaluation: report plain CE so eval_loss is comparable across ALL models
        # (teacher, baseline, distilled). The teacher is not needed here.
        if not model.training:
            out = SequenceClassifierOutput(loss=ce_loss, logits=student_out.logits)
            return (ce_loss, out) if return_outputs else ce_loss
 
        # (b) Teacher forward on the same batch: no labels, no gradients.
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            teacher_out = self.teacher(**teacher_inputs, output_hidden_states=use_hidden)
 
        # (c) KD loss. Dividing logits by T > 1 softens both distributions, so the
        #     student also sees how the teacher ranks the wrong class; T^2 restores the
        #     gradient scale (soft-target gradients shrink by 1/T^2) so CE and KD stay
        #     balanced (Hinton et al., 2015).
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_out.logits / T, dim=-1),
            F.softmax(teacher_out.logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
 
        loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
 
        # (d) Optional: align the student's last hidden state with the teacher's
        #     (cosine, real tokens only; padding is masked out). Both are 768-dimensional.
        if use_hidden:
            mask = inputs["attention_mask"].bool()
            s_hidden = student_out.hidden_states[-1][mask]
            t_hidden = teacher_out.hidden_states[-1][mask]
            target = torch.ones(s_hidden.size(0), device=s_hidden.device)
            loss = loss + self.beta * F.cosine_embedding_loss(s_hidden, t_hidden, target)
 
        self._ce_sum += ce_loss.detach()
        self._kd_sum += kd_loss.detach()
        self._steps += 1
 
        return (loss, student_out) if return_outputs else loss
 
    def log(self, logs, *args, **kwargs):
        # Add the separate CE / KD averages to every training log line.
        if "loss" in logs and self._steps > 0:
            logs["ce_loss"] = round((self._ce_sum / self._steps).item(), 4)
            logs["kd_loss"] = round((self._kd_sum / self._steps).item(), 4)
            self._ce_sum.zero_()
            self._kd_sum.zero_()
            self._steps = 0
        super().log(logs, *args, **kwargs)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="scratch", choices=["scratch", "pretrained"])
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--teacher_dir", default=TEACHER_DIR)
    args = parser.parse_args()
    lr = args.lr or STUDENT_DEFAULTS[args.init]["lr"]
    epochs = args.epochs or STUDENT_DEFAULTS[args.init]["epochs"]
    output_dir = student_dir("distill", args.init, args.layers)
 
    dataset, tokenizer = load_sst2()
    teacher = build_teacher(args.teacher_dir)       # already fine-tuned in step 1
    student = build_student(args.init, args.layers)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
    trainer = DistillationTrainer(
        model=student,
        args=make_training_args(output_dir, epochs=epochs, lr=lr),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
    )
 
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_history(trainer, output_dir)
 
    metrics = trainer.evaluate()
    print(f"[student_distill_{args.init}_{args.layers}L] accuracy={metrics['eval_accuracy']:.4f} "
          f"val_ce_loss={metrics['eval_loss']:.4f} "
          f"params={count_parameters(student) / 1e6:.1f}M lr={lr} epochs={epochs} "
          f"T={args.temperature} alpha={args.alpha} beta={args.beta}")
 
 
if __name__ == "__main__":
    main()
 
# -----------------------------------------------------------------------------
# USAGE & RATIONALE
# * KL divergence with temperature (instead of MSE on logits, or hard labels only): it
#   compares full probability distributions, and the soft targets carry "dark knowledge"
#   (how confident the teacher is, and how similar the classes look to it) that a 0/1
#   label cannot. See README section 2 for the history and motivation.
# * alpha mixes the true label (protects the student from teacher mistakes) with the
#   teacher's targets.
# * Hidden-state cosine loss (DistilBERT's third loss) is only possible because teacher
#   and student have the same hidden size. It is off by default; a from-scratch student
#   often profits from it, so try --beta 1.
# * Evaluation uses plain CE for every model, so val_ce_loss is directly comparable.
#   During TRAINING the log shows loss (combined), ce_loss and kd_loss separately.
# * Tuning to get closest to the teacher (change one setting at a time): temperature in
#   {1, 2, 4}, alpha in {0.3, 0.5, 0.7}, beta in {0, 1}, lr in {5e-5, 1e-4, 2e-4},
#   epochs in {10, 15}.