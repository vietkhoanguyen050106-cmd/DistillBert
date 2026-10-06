import os
 
# ----- Models ----------------------------------------------------------------
TEACHER_NAME = "bert-base-uncased"          # 12 layers, 768 hidden, ~109.5M params (with head)
STUDENT_NAME = "distilbert-base-uncased"    # 6 layers, 768 hidden, ~67.0M params (with head)
TOKENIZER_NAME = "bert-base-uncased"        # DistilBERT uses exactly the BERT-uncased vocabulary
NUM_LABELS = 2                              # SST-2: 0 = negative, 1 = positive
MAX_LENGTH = 128
SEED = 42
BATCH_SIZE = 8
 
# ----- Learning rate / epochs ------------------------------------------------
# A pretrained model is only *adjusted*, so it needs a small learning rate (and few
# epochs) to avoid destroying what it already knows. A model trained from random
# weights has to learn everything, so it needs a larger learning rate and more epochs.
TEACHER_LR = 2e-5
TEACHER_EPOCHS = 3
STUDENT_DEFAULTS = {
    "scratch":    {"lr": 1e-4, "epochs": 10},   # random weights, DistilBERT architecture
    "pretrained": {"lr": 5e-5, "epochs": 3},    # distilbert-base-uncased weights (reference run)
}
 
# ----- Distillation hyper-parameters ----------------------------------------
TEMPERATURE = 2.0   # T: softens the teacher's probabilities
ALPHA = 0.5         # weight of the hard-label loss; (1 - ALPHA) is the weight of the teacher loss
BETA = 1.0          # weight of the optional hidden-state (cosine) loss; 0 = disabled
 
# ----- Output locations -----------------------------------------------------
OUTPUT_ROOT = "./outputs"
TEACHER_DIR = os.path.join(OUTPUT_ROOT, "teacher")
 
 
def student_dir(mode, init, num_layers):
    """mode: 'baseline' (hard labels only) or 'distill'; init: 'scratch' or 'pretrained'."""
    return os.path.join(OUTPUT_ROOT, f"student_{mode}_{init}_{num_layers}L")
 
 
def is_trained(directory):
    """A model counts as finished once save_model() wrote config.json in its folder."""
    return os.path.exists(os.path.join(directory, "config.json"))