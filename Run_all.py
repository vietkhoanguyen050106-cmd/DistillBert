import argparse
import subprocess
import sys
 
from config import TEACHER_DIR, is_trained, student_dir
 
 
def run(module, *extra):
    command = [sys.executable, "-m", module, *extra]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, check=True)      # one process per stage frees GPU memory
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[6])   
    parser.add_argument("--inits", nargs="+", default=["scratch"],
                        choices=["scratch", "pretrained"])
    args = parser.parse_args()
 
    if not is_trained(TEACHER_DIR):
        run("Bert.train_teacher")
 
    for init in args.inits:
        sizes = args.layers if init == "scratch" else [6]   # pretrained DistilBERT = 6 layers
        for layers in sizes:
            extra = ["--init", init, "--layers", str(layers)]
            if not is_trained(student_dir("baseline", init, layers)):
                run("Bert.train_student", *extra)
            if not is_trained(student_dir("distill", init, layers)):
                run("Bert.train_distill", *extra)
 
    run("Bert.benchmark")
    run("Bert.plot_curves")
 
 
if __name__ == "__main__":
    main()
 
# USAGE & RATIONALE
# * Finished models are skipped, so a stopped Kaggle session can be resumed (as long
#   as ./outputs still exists).
# * Several student sizes answer "how small can the network become?"; the pretrained
#   run is an optional reference showing what pretraining adds on top of distillation.