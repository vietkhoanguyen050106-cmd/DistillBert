from transformers import AutoModelForSequenceClassification
 
from config import NUM_LABELS, TEACHER_NAME
 
 
def build_teacher(path=None):
    """
    path=None  -> pretrained bert-base-uncased from Hugging Face (new random
                  classification head; must be fine-tuned with train_teacher.py).
    path="dir" -> a teacher already fine-tuned and saved in `dir`.
 
    Usage:
        teacher = build_teacher()
        teacher = build_teacher("./outputs/teacher")
    """
    return AutoModelForSequenceClassification.from_pretrained(
        path or TEACHER_NAME, num_labels=NUM_LABELS
    )
    
# USAGE & RATIONALE
# * The teacher is a pretrained model that is FINE-TUNED on SST-2: BERT already
#   understands English from pretraining, fine-tuning only teaches it the sentiment
#   task. That is why a small learning rate (2e-5) and few epochs (3) are used.
# * "Some weights ... newly initialized" in the log refers to the new classifier head
#   and is expected.