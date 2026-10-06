import csv
import glob
import json
import os
import time
 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
 
from Bert.config import OUTPUT_ROOT, TEACHER_DIR
from Bert.dataset import load_sst2
from Bert.utils import count_parameters
 
 
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
 
 
@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    outputs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs.append(model(**batch).logits.float().cpu())
    return torch.cat(outputs)
 
 
@torch.no_grad()
def measure_latency(model, tokenizer, device, runs=100, warmup=10):
    """Average milliseconds to classify ONE sentence (batch size 1, 64 tokens)."""
    model.eval()
    sample = tokenizer(
        "a gorgeous, witty and moving film",
        padding="max_length", max_length=64, return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)
    for _ in range(warmup):                 # warm-up: exclude one-off start-up cost
        model(**sample)
    sync(device)
    start = time.perf_counter()
    for _ in range(runs):
        model(**sample)
    sync(device)
    return (time.perf_counter() - start) / runs * 1000
 
 
def evaluate_model(model_dir, loader, labels, tokenizer, device, teacher_logits=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    params = count_parameters(model)
 
    sync(device)
    start = time.perf_counter()
    logits = predict_logits(model, loader, device)      # whole validation set
    sync(device)
    elapsed = time.perf_counter() - start
 
    predictions = logits.argmax(dim=-1)
    row = {
        "model": os.path.basename(model_dir.rstrip("/")),
        "layers": model.config.num_hidden_layers,
        "params_M": round(params / 1e6, 1),
        "size_MB": round(params * 4 / 1e6, 1),          # float32 = 4 bytes per parameter
        "accuracy": round((predictions == labels).float().mean().item() * 100, 2),
        "val_ce_loss": round(F.cross_entropy(logits, labels).item(), 4),
        "latency_ms": round(measure_latency(model, tokenizer, device), 2),
        "throughput_sent_per_s": round(len(labels) / elapsed, 1),
    }
    if teacher_logits is not None:
        row["agreement_with_teacher"] = round(
            (predictions == teacher_logits.argmax(dim=-1)).float().mean().item() * 100, 2
        )
        # KL(teacher || student) at T = 1, averaged over the validation sentences
        row["kl_to_teacher"] = round(
            F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(teacher_logits, dim=-1),
                     reduction="batchmean").item(), 4
        )
    return row, logits
 
 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, tokenizer = load_sst2()
    validation = dataset["validation"]
    labels = torch.tensor(validation["label"])
    loader = DataLoader(
        validation, batch_size=64,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
 
    # Teacher first: every student is measured relative to it.
    teacher_row, teacher_logits = evaluate_model(TEACHER_DIR, loader, labels, tokenizer, device)
    teacher_row["agreement_with_teacher"] = 100.0
    teacher_row["kl_to_teacher"] = 0.0
    rows = [teacher_row]
 
    student_dirs = sorted(
        d for d in glob.glob(os.path.join(OUTPUT_ROOT, "student_*"))
        if os.path.exists(os.path.join(d, "config.json"))
    )
    for directory in student_dirs:
        row, _ = evaluate_model(directory, loader, labels, tokenizer, device, teacher_logits)
        rows.append(row)
 
    for row in rows:
        row["accuracy_gap_vs_teacher"] = round(row["accuracy"] - teacher_row["accuracy"], 2)
        row["compression_x"] = round(teacher_row["params_M"] / row["params_M"], 2)
        row["speedup_x"] = round(teacher_row["latency_ms"] / row["latency_ms"], 2)
 
    columns = ["model", "layers", "params_M", "accuracy", "accuracy_gap_vs_teacher",
               "val_ce_loss", "agreement_with_teacher", "kl_to_teacher",
               "latency_ms", "speedup_x", "compression_x"]
    print("\n" + " | ".join(columns))
    for row in rows:
        print(" | ".join(str(row.get(c, "")) for c in columns))
 
    with open(os.path.join(OUTPUT_ROOT, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    all_columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(os.path.join(OUTPUT_ROOT, "comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
 
    try:                                            # plot: accuracy vs size
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        plt.figure(figsize=(8, 5))
        for row in rows:
            name = row["model"]
            marker = "*" if name == "teacher" else ("o" if "distill" in name else "s")
            plt.scatter(row["params_M"], row["accuracy"], marker=marker, s=90)
            plt.annotate(name, (row["params_M"], row["accuracy"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
        plt.xlabel("Parameters (millions)")
        plt.ylabel("SST-2 validation accuracy (%)")
        plt.title("Model size vs accuracy")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, "comparison.png"), dpi=150)
    except Exception as error:                      # plotting must never break the run
        print("plot skipped:", error)
 
 
if __name__ == "__main__":
    main()
 

# USAGE & RATIONALE
# * Accuracy alone cannot answer "how much did the student learn from the teacher?".
#   Three complementary numbers do:
#     - accuracy_gap_vs_teacher : how far below the teacher the student ends
#     - agreement_with_teacher  : % of sentences where both predict the same label
#     - kl_to_teacher           : how different their probability outputs are
#       (0 = identical). Distillation explicitly minimises this quantity.
# * val_ce_loss is the plain cross-entropy on the validation set for every model, so
#   the "loss" column can be compared directly.
# * Latency: batch size 1, fixed 64 tokens, after warm-up, with cuda synchronize before
#   reading the clock (GPU calls are asynchronous). Absolute values depend on the GPU;
#   the ratio speedup_x is what to report.
# * Size in MB assumes float32 weights (4 bytes per parameter).