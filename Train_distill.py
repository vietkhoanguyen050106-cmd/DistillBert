import argparse

import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

from Bert.config import (
    ALPHA,
    BETA,
    STUDENT_DEFAULTS,
    TEACHER_DIR,
    TEMPERATURE,
    student_dir,
)
from Bert.dataset import load_sst2
from Bert.Model_student import build_student
from Bert.Model_teacher import build_teacher
from Bert.utils import (
    compute_metrics,
    count_parameters,
    make_training_args,
    save_history,
)


class DistillationTrainer(Trainer):
    """Hugging Face Trainer with teacher-student distillation loss."""

    def __init__(
        self,
        *args,
        teacher,
        temperature,
        alpha,
        beta,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.teacher = teacher.to(self.args.device)
        self.teacher.eval()

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

        # The custom loss does not use num_items_in_batch.
        self.model_accepts_loss_kwargs = False

        # Running sums for logging.
        self._ce_sum = torch.zeros(
            (),
            device=self.args.device,
        )

        self._kd_sum = torch.zeros(
            (),
            device=self.args.device,
        )

        self._cos_sum = torch.zeros(
            (),
            device=self.args.device,
        )

        self._steps = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        use_hidden = (
            self.beta > 0
            and model.training
        )

        # ---------------------------------------------------------
        # 1. Student forward pass
        # ---------------------------------------------------------
        student_out = model(
            **inputs,
            output_hidden_states=use_hidden,
        )

        ce_loss = student_out.loss.mean()

        # During evaluation, use only the normal CE loss.
        # This makes eval_loss comparable with Teacher and baseline Student.
        if not model.training:
            out = SequenceClassifierOutput(
                loss=ce_loss,
                logits=student_out.logits,
            )

            return (
                (ce_loss, out)
                if return_outputs
                else ce_loss
            )

        # ---------------------------------------------------------
        # 2. Teacher forward pass
        # ---------------------------------------------------------
        teacher_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "labels"
        }

        with torch.no_grad():
            teacher_out = self.teacher(
                **teacher_inputs,
                output_hidden_states=use_hidden,
            )

        # ---------------------------------------------------------
        # 3. Logit-based knowledge distillation
        # ---------------------------------------------------------
        temperature = self.temperature

        kd_loss = F.kl_div(
            F.log_softmax(
                student_out.logits / temperature,
                dim=-1,
            ),
            F.softmax(
                teacher_out.logits / temperature,
                dim=-1,
            ),
            reduction="batchmean",
        ) * (temperature ** 2)

        # Combined CE + KD objective.
        loss = (
            self.alpha * ce_loss
            + (1.0 - self.alpha) * kd_loss
        )

        # ---------------------------------------------------------
        # 4. Hidden / feature-based distillation
        # ---------------------------------------------------------
        cos_loss = torch.zeros(
            (),
            device=student_out.logits.device,
        )

        if use_hidden:
            attention_mask = inputs["attention_mask"].bool()

            student_hidden = (
                student_out.hidden_states[-1][attention_mask]
            )

            teacher_hidden = (
                teacher_out.hidden_states[-1][attention_mask]
            )

            target = torch.ones(
                student_hidden.size(0),
                device=student_hidden.device,
            )

            cos_loss = F.cosine_embedding_loss(
                student_hidden,
                teacher_hidden,
                target,
            )

            loss = (
                loss
                + self.beta * cos_loss
            )

        # ---------------------------------------------------------
        # 5. Store individual loss components for logging
        # ---------------------------------------------------------
        self._ce_sum += ce_loss.detach()
        self._kd_sum += kd_loss.detach()
        self._cos_sum += cos_loss.detach()
        self._steps += 1

        return (
            (loss, student_out)
            if return_outputs
            else loss
        )

    def log(self, logs, *args, **kwargs):
        """Add CE, KD and cosine feature losses to Trainer logs."""

        if "loss" in logs and self._steps > 0:
            logs["ce_loss"] = round(
                (
                    self._ce_sum
                    / self._steps
                ).item(),
                4,
            )

            logs["kd_loss"] = round(
                (
                    self._kd_sum
                    / self._steps
                ).item(),
                4,
            )

            logs["cos_loss"] = round(
                (
                    self._cos_sum
                    / self._steps
                ).item(),
                4,
            )

            self._ce_sum.zero_()
            self._kd_sum.zero_()
            self._cos_sum.zero_()

            self._steps = 0

        super().log(
            logs,
            *args,
            **kwargs,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init",
        default="scratch",
        choices=["scratch", "pretrained"],
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=BETA,
    )

    parser.add_argument(
        "--teacher_dir",
        default=TEACHER_DIR,
    )

    args = parser.parse_args()

    lr = (
        args.lr
        or STUDENT_DEFAULTS[args.init]["lr"]
    )

    epochs = (
        args.epochs
        or STUDENT_DEFAULTS[args.init]["epochs"]
    )

    output_dir = student_dir(
        "distill",
        args.init,
        args.layers,
    )

    dataset, tokenizer = load_sst2()

    # The Teacher is already fine-tuned.
    teacher = build_teacher(
        args.teacher_dir
    )

    # The Student is initialized from scratch
    # when --init scratch is used.
    student = build_student(
        args.init,
        args.layers,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    trainer = DistillationTrainer(
        model=student,
        args=make_training_args(
            output_dir,
            epochs=epochs,
            lr=lr,
        ),
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

    trainer.save_model(
        output_dir
    )

    tokenizer.save_pretrained(
        output_dir
    )

    save_history(
        trainer,
        output_dir,
    )

    metrics = trainer.evaluate()

    print(
        f"[student_distill_{args.init}_{args.layers}L] "
        f"accuracy={metrics['eval_accuracy']:.4f} "
        f"val_ce_loss={metrics['eval_loss']:.4f} "
        f"params={count_parameters(student) / 1e6:.1f}M "
        f"lr={lr} "
        f"epochs={epochs} "
        f"T={args.temperature} "
        f"alpha={args.alpha} "
        f"beta={args.beta}"
    )


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# RATIONALE
#
# 1. Cross-Entropy Loss
#    The ground-truth SST-2 label provides direct task supervision.
#
# 2. KL-Divergence Distillation Loss
#    The Teacher provides soft probability targets.
#    Temperature softens the distribution so the Student can learn
#    the Teacher's relative confidence between classes.
#
# 3. Temperature^2
#    Compensates for the gradient scaling introduced by temperature.
#
# 4. Hidden / Feature Distillation
#    The Student is also encouraged to produce hidden representations
#    with similar directions to the Teacher's final hidden representations.
#
# 5. Cosine Embedding Loss
#    Cosine similarity compares representation direction rather than
#    requiring identical vector magnitudes.
#
# 6. Padding Mask
#    Padding tokens are excluded from the feature loss because PAD tokens
#    do not carry sentence semantics.
#
# 7. Teacher is frozen
#    The Teacher provides a fixed learning target.
#    Only the Student parameters are updated.
#
# 8. Evaluation uses plain CE
#    This keeps eval_loss directly comparable between Teacher,
#    Student baseline and Student with distillation.
# -----------------------------------------------------------------------------