import evaluate
import numpy as np
import torch
from transformers import TrainingArguments
 
from config import BATCH_SIZE, SEED
 
_accuracy = evaluate.load("accuracy")
 
 
def compute_metrics(eval_pred):
    """Called by Trainer at every evaluation: accuracy on the validation set."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return _accuracy.compute(predictions=predictions, references=labels)
 
 
def make_training_args(output_dir, epochs, lr):
    """
    One training configuration for ALL models; only epochs and lr are passed in.
      weight_decay=0.01        : light regularisation
      warmup_ratio=0.1         : lr ramps up during the first 10% of steps and then decays
                                 linearly to 0 (stabilises BERT-style models)
      eval/save every 500 steps: ~dense learning curves; load_best_model_at_end keeps the
                                 checkpoint with the best validation accuracy
      save_total_limit=1       : save disk space
      fp16                     : mixed precision on GPU
      seed                     : reproducibility
    Gradient clipping at 1.0 is the Trainer default.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )
 
 
def count_parameters(model):
    """Total number of parameters (the 'size' axis of the trade-off)."""
    return sum(p.numel() for p in model.parameters())
 
 
def save_history(trainer, output_dir):
    """Write loss / accuracy / learning-rate logs to log_history.json for plot_curves.py."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)