from transformers import AutoConfig, AutoModelForSequenceClassification, set_seed
 
from Bert.config import NUM_LABELS, SEED, STUDENT_NAME
 
 
def build_student(init="scratch", num_layers=6):
    """
    init="scratch"    -> DistilBERT ARCHITECTURE with RANDOM weights. The student knows
                         nothing except what SST-2 (and the teacher) teach it.
                         `num_layers` can be reduced (e.g. 4 or 2) to study
                         "how small can the network become?".
    init="pretrained" -> distilbert-base-uncased with its pretrained weights
                         (reference run; always 6 layers).
 
    Usage:
        student = build_student("scratch", num_layers=6)
    """
    if init == "pretrained":
        if num_layers != 6:
            raise ValueError("the pretrained DistilBERT checkpoint has exactly 6 layers")
        return AutoModelForSequenceClassification.from_pretrained(
            STUDENT_NAME, num_labels=NUM_LABELS
        )
 
    if init == "scratch":
        set_seed(SEED)   # same random starting weights for the baseline and the distilled run
        config = AutoConfig.from_pretrained(STUDENT_NAME, num_labels=NUM_LABELS)
        config.n_layers = num_layers
        return AutoModelForSequenceClassification.from_config(config)
 
    raise ValueError(f"unknown init '{init}' (use 'scratch' or 'pretrained')")

# USAGE & RATIONALE
# * from_config (instead of from_pretrained) is what makes the student really "from
#   scratch": it builds the network and initialises it randomly, nothing is downloaded.
# * DistilBERT as the student architecture: it is the well-known 6-layer version of
#   BERT made by distillation (Sanh et al., 2019), has the same hidden size and
#   vocabulary as the teacher, and is supported by the same Hugging Face classes.
# * Fixing the seed before building makes the baseline and the distilled student start
#   from IDENTICAL weights, so any difference between them comes from the loss only.
# * Expectation (not a measurement): a transformer trained from random weights on only
#   ~67k short SST-2 examples lacks the language knowledge BERT gets from pretraining,
#   so it will usually end clearly below the teacher. Distillation is expected to help
#   it more than it helps a pretrained student, because the teacher supplies extra
#   information the tiny dataset cannot.